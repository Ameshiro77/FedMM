import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *
from FLAlgorithms.config import *
from FLAlgorithms.trainmodel.losses import *

class UserNoDis(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1, args=None):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)

        # === 配置参数 ===
        self.conf_threshold = 0.9  
        
        self.client_align_weight = args.client_align_weight
        self.client_reg_weight = args.client_reg_weight
        self.client_logits_weight = args.client_logits_weight
        
        self.args = args

    def upload_shares(self):
        """
        上传特征 z
        """
        self.ae_model.eval()
        z_shares_dict = {m: None for m in self.modalities}

        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}

        for m in self.modalities:
            x_m = torch.from_numpy(X_modal[m]).to(self.device)
            
            # === 【修改点】encode 返回2个值 (out, out1) ===
            with torch.no_grad():
                z, _ = self.ae_model.encode(x_m, m)
            
            # 平均池化 [B, T, D] -> [B, D]
            z_mean = z.mean(dim=1)
            z_shares_dict[m] = z_mean.detach().cpu()

        return z_shares_dict

    def train_ae_prop(self, global_share, global_prototypes, pre_w):
        """
        [消融版] 自编码器训练
        """
        self.ae_model.train()
        self.cf_model.eval()

        z_shares_all = {m: [] for m in self.modalities}
        loss_dict_epoch = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}

        for ep in range(self.local_epochs):
            modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer_ae.zero_grad()
                rec_loss, align_loss, reg_loss = 0.0, 0.0, 0.0
                
                orth_loss = 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    
                    # === 【修改点】分开调用 forward 和 encode ===
                    # 1. 拿重构结果 (forward 只返回 x_recon)
                    x_recon = self.ae_model(x_m, m)
                    
                    # 2. 拿特征 (encode 返回 out, out1)
                    # 我们需要 out (序列特征) 来做 z
                    z, _ = self.ae_model.encode(x_m, m)

                    z_shares_all[m].append(z.detach().cpu())

                    # 1. 重构损失
                    rec_loss += self.rec_loss_fn(x_recon, x_m)

                    # 2. 对齐损失 (z vs global_z)
                    if global_share is not None:
                        # global_share 是 [D] 或 [1, D]
                        z_pooled = z.mean(dim=1) # [B, D]
                        z_global_srv = global_share.detach().to(self.device)
                        
                        cos_sim = F.cosine_similarity(z_pooled, z_global_srv, dim=1)
                        align_loss_term = 1 - cos_sim.mean()
                        align_loss += align_loss_term

                    # 3. 正则化
                    if pre_w is not None:
                        reg = 0.0
                        for p_new, p_old in zip(self.ae_model.parameters(), pre_w):
                            reg += torch.sum((p_new - p_old.to(self.device))**2)
                        reg_loss += reg

                # === 总损失 ===
                loss = rec_loss + \
                       self.client_align_weight * align_loss + \
                       self.client_reg_weight * reg_loss

                loss.backward()
                self.optimizer_ae.step()

                loss_dict_epoch['rec_loss'].append(rec_loss.item())
                loss_dict_epoch['orth_loss'].append(0.0)
                loss_dict_epoch['align_loss'].append(align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss)
                loss_dict_epoch['reg_loss'].append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)

        loss_dict_avg = {k: np.mean(v) for k, v in loss_dict_epoch.items()}
        return loss_dict_avg 


    def train_cf_prop(self, server_logits_bank=None, base_alpha=0.5, T=3.0):
        """
        [消融版] 分类器训练
        """
        self.freeze(self.ae_model)
        self.cf_model.train()

        loss_dict_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
            modalities_seq, y_seq = make_seq_batch2(self.labeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer_cf.zero_grad()
                latents = []

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    # === 【修改点】encode 返回 2 个值 ===
                    z, _ = self.ae_model.encode(x_m, m)
                    latents.append(z)

                latent_cat = torch.cat(latents, dim=-1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()

                logits = self.cf_model(latent_cat)

                # === 1. CE Loss ===
                ce_loss = self.cls_loss_fn(logits, y_batch)

                # === 2. 蒸馏 ===
                kd_loss = 0.0
                alpha = 0.0 
                
                if server_logits_bank is not None:
                     teacher_logits = []
                     y_np = y_batch.detach().cpu().numpy()
                     has_teacher = True
                     for y in y_np:
                         if y not in server_logits_bank:
                             has_teacher = False
                             break
                         teacher_logits.append(server_logits_bank[y].unsqueeze(0))
                     
                     if has_teacher:
                         teacher_logits = torch.cat(teacher_logits, dim=0).to(self.device)
                         alpha = base_alpha
                         p_s = F.log_softmax(logits / T, dim=1)
                         p_t = F.softmax(teacher_logits / T, dim=1)
                         kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

                loss = (1 - alpha) * ce_loss + alpha * kd_loss

                loss.backward()
                self.optimizer_cf.step()

                pred = torch.argmax(logits, dim=-1)
                correct = (pred == y_batch).sum().item()
                loss_dict_epoch['cf_loss'].append(loss.item())
                loss_dict_epoch['cf_acc'].append(correct / y_batch.size(0))

        self.unfreeze(self.ae_model)

        return {
            'cf_loss': np.mean(loss_dict_epoch['cf_loss']),
            'cf_acc': np.mean(loss_dict_epoch['cf_acc'])
        }

    def test(self, eval_win=EVAL_WIN): 
        self.ae_model.eval()
        self.cf_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        y_test = self.test_data["y"]
        seq_len = len(y_test)

        with torch.no_grad():
            for start in range(0, seq_len - eval_win, eval_win):
                end = start + eval_win

                batch_x = {}
                for m in self.modalities:
                    d = self.test_data[m][start:end]
                    batch_x[m] = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(self.device)

                y_batch = torch.tensor(y_test[start:end], dtype=torch.long).to(self.device)
                y_batch = y_batch.view(-1)

                latents = []
                for m in self.modalities:
                    # === 【修改点】encode 返回 2 个值 ===
                    z, _ = self.ae_model.encode(batch_x[m], m)
                    latents.append(z)

                if len(latents) > 0:
                    latent_cat = torch.cat(latents, dim=-1) 
                    logits = self.cf_model(latent_cat)

                    if logits.dim() == 3:
                        logits = logits.view(-1, logits.shape[-1])

                    loss = self.cls_loss_fn(logits, y_batch)
                    total_loss += loss.item() * y_batch.size(0)

                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return avg_acc, avg_loss