import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import make_seq_batch2

class UserGao(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio, args):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio)
        
        self.conf_threshold = getattr(args, 'conf_thresh', 0.8) 
        self.unc_threshold = getattr(args, 'unc_thresh', 0.1)   
        self.args = args
        
        # === 改造：本地历史模型池 ===
        self.history_pool = [] 
        self.max_history_size = 2 # 存过去 2 个模型即可，多了显存爆炸且没必要

    def update_history_pool(self):
        """ 将当前模型状态存入历史池 """
        # 深拷贝当前状态
        current_state = {
            'ae': copy.deepcopy(self.ae_model.state_dict()),
            'cf': copy.deepcopy(self.cf_model.state_dict())
        }
        
        self.history_pool.append(current_state)
        # 保持池子大小
        if len(self.history_pool) > self.max_history_size:
            self.history_pool.pop(0) # 移除最旧的

    def train_ae(self):
        """ 阶段1: 无监督重构 (不变) """
        self.ae_model.train()
        self.cf_model.eval()
        rec_loss_lst = []

        for _ in range(self.local_epochs):
            modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]
            
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer_ae.zero_grad()
                rec_loss = 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)

                rec_loss.backward()
                self.optimizer_ae.step()
                rec_loss_lst.append(rec_loss.item())

        return np.mean(rec_loss_lst)

    def personalize(self):
        """ 
        阶段2: 改造版个性化 (Temporal Self-Ensembling) 
        不需要外部传入模型，完全依赖本地历史 + 当前全局
        """
        # 如果没有历史模型 (第一轮)，先存一个然后跳过，或者直接基于置信度训练
        if not self.history_pool:
            self.update_history_pool()
            return

        self.ae_model.train()
        self.cf_model.train()
        
        for _ in range(self.local_epochs):
            modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]
            
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                # 准备 Batch
                batch_x = {}
                for m in self.modalities:
                    batch_x[m] = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)

                # =========================================================
                # 步骤 1: 生成伪标签 (利用当前模型 + 历史模型)
                # =========================================================
                with torch.no_grad():
                    # 1. 当前模型预测 [B, T, C]
                    curr_probs = self._get_probs(self.ae_model, self.cf_model, batch_x)
                    
                    # 2. 历史模型预测 (Auxiliary)
                    history_probs_list = []
                    
                    # 备份当前参数
                    temp_ae_state = copy.deepcopy(self.ae_model.state_dict())
                    temp_cf_state = copy.deepcopy(self.cf_model.state_dict())
                    
                    # 遍历本地历史池
                    for state in self.history_pool:
                        self.ae_model.load_state_dict(state['ae'])
                        self.cf_model.load_state_dict(state['cf'])
                        self.ae_model.eval(); self.cf_model.eval()
                        
                        prob = self._get_probs(self.ae_model, self.cf_model, batch_x)
                        history_probs_list.append(prob)
                    
                    # 恢复参数进行训练
                    self.ae_model.load_state_dict(temp_ae_state)
                    self.cf_model.load_state_dict(temp_cf_state)
                    self.ae_model.train(); self.cf_model.train()
                    
                    # 3. 计算不确定性 (Uncertainty)
                    # Ensemble = Current + History
                    all_probs = torch.stack([curr_probs] + history_probs_list) # [N+1, B, T, C]
                    std_probs = torch.std(all_probs, dim=0) # [B, T, C]
                    
                    # 获取当前预测 (Pseudo Label)
                    max_probs, pseudo_labels = torch.max(curr_probs, dim=-1) # [B, T]
                    
                    # 获取对应类别的 Uncertianty
                    uncertainty = torch.gather(std_probs, -1, pseudo_labels.unsqueeze(-1)).squeeze(-1) # [B, T]
                    
                    # 4. 生成掩码 (Filter)
                    mask = (max_probs >= self.conf_threshold) & (uncertainty <= self.unc_threshold)
                    
                # =========================================================
                # 步骤 2: 微调
                # =========================================================
                if mask.sum() == 0:
                    continue
                
                self.optimizer_ae.zero_grad()
                self.optimizer_cf.zero_grad()
                
                latents = []
                for m in self.modalities:
                    _, z = self.ae_model.encode(batch_x[m], m) 
                    latents.append(z)
                logits = self.cf_model(torch.cat(latents, dim=-1))
                
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = pseudo_labels.reshape(-1)
                mask_flat = mask.reshape(-1).float()
                
                loss_raw = F.cross_entropy(logits_flat, labels_flat, reduction='none')
                loss = (loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                loss.backward()
                self.optimizer_ae.step()
                self.optimizer_cf.step()
        
        # 个性化结束后，将当前更强的模型加入历史池，供下一轮使用
        self.update_history_pool()

    def _get_probs(self, ae, cf, x_dict):
        latents = []
        for m in self.modalities:
            _, z = ae.encode(x_dict[m], m) 
            latents.append(z)
        logits = cf(torch.cat(latents, dim=-1))
        return F.softmax(logits, dim=-1)