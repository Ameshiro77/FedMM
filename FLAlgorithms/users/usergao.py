import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *

class UserGao(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3, beta=1.0, lamda=0.0, 
                 local_epochs=1, label_ratio=0.1):
        
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio)

        # 确保 modalities 是列表
        if not isinstance(self.modalities, list):
            self.modalities = [self.modalities]

        # 优化器：分开定义
        # 1. 仅优化 AE (用于 Phase 1: Unsupervised)
        self.optimizer_ae = torch.optim.Adam(self.ae_model.parameters(), lr=self.learning_rate)
        
        # 2. 同时优化 AE + CF (用于 Phase 3: Personalization)
        # 论文中微调通常使用较小的学习率，这里设为 0.001 或更小
        self.optimizer_total = torch.optim.Adam(
            list(self.ae_model.parameters()) + list(self.cf_model.parameters()),
            lr=self.learning_rate * 0.1 
        )
        
        self.rec_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # [cite_start]伪标签阈值 (论文公式 1 中的 gamma [cite: 132])
        self.conf_threshold = 0.8 

    def train_ae(self):
        """
        [Phase 1] 纯无监督自编码器训练：
        利用本地无标签数据，优化 AE 的重构能力。
        """
        self.ae_model.train()
        self.cf_model.eval() 
        
        epoch_loss = []
        
        for ep in range(self.local_epochs):
            # 获取本地数据 (假设全部无标签)
            modalities_seq, _ = make_seq_batch2(self.train_data, self.batch_size)
            X_local = {m: modalities_seq[m] for m in self.modalities}
            
            # 获取序列长度
            first_modal = self.modalities[0]
            seq_len = X_local[first_modal].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                # 随机滑窗
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                self.optimizer_ae.zero_grad()
                total_loss = 0.0
                
                # 对每个模态计算重构损失
                for m in self.modalities:
                    x_seg = torch.from_numpy(X_local[m][:, idx_start:idx_end, :]).float().to(self.device)
                    
                    # SplitLSTMAutoEncoder forward 返回的是 x_recon
                    x_recon = self.ae_model(x_seg, m)
                    
                    # MSE Loss
                    loss = self.rec_loss_fn(x_recon, x_seg)
                    total_loss += loss
                
                total_loss.backward()
                self.optimizer_ae.step()
                epoch_loss.append(total_loss.item())
                
        return np.mean(epoch_loss) if len(epoch_loss) > 0 else 0.0

    def train_personalization(self):
        """
        [Phase 3] 基于伪标签的个性化微调：
        先生成伪标签，筛选高置信度样本，再进行有监督微调。
        """
        # === Step 1: 生成伪标签 (Pseudo-labeling) ===
        self.ae_model.eval()
        self.cf_model.eval()
        
        # 获取所有本地数据用于打标
        # 注意：为了代码简单，这里直接一次性取全量数据（如果显存不够，需要分批处理）
        # 假设 self.train_data['y'] 长度即样本数
        total_samples = len(self.train_data['y'])
        full_seq, _ = make_seq_batch2(self.train_data, total_samples) # batch_size = total
        
        X_full = {m: torch.from_numpy(full_seq[m]).float().to(self.device) for m in self.modalities}
        
        pseudo_labels_dict = {} # 存储 (batch_idx, time_idx) -> label
        valid_indices = []      # 存储高置信度的索引位置
        
        with torch.no_grad():
            # 滑窗遍历整个长序列进行预测
            # 为了效率，这里使用较大的固定窗口步进
            step_size = 32
            seq_len_total = X_full[self.modalities[0]].shape[1]
            
            for t in range(0, seq_len_total - step_size, step_size):
                reps_list = []
                for m in self.modalities:
                    x_seg = X_full[m][:, t:t+step_size, :] # [B, Win, D]
                    # 提取特征
                    _, out = self.ae_model.encoders[m](x_seg)
                    reps_list.append(out[:, -1, :]) # [B, Rep]
                
                rep_cat = torch.cat(reps_list, dim=-1)
                logits = self.cf_model(rep_cat)
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1) # [B]
                
                # 筛选
                mask = max_probs > self.conf_threshold
                
                # 记录有效的样本信息
                # 这里简单处理：我们将在下面的训练循环中重新计算并应用mask，
                # 为了保持 consistency，这里只是验证是否有样本可用
                if mask.sum() > 0:
                    valid_indices.append(t)

        if len(valid_indices) == 0:
            return 0.0 # 没有样本满足置信度，跳过微调

        # === Step 2: 使用伪标签进行微调 ===
        self.ae_model.train()
        self.cf_model.train()
        epoch_loss = []
        
        # 重新分 Batch 训练
        batch_size = self.batch_size
        # 这里使用标准的随机滑窗训练逻辑，但在计算 Loss 时应用 Mask
        
        # 重新获取数据流
        modalities_seq, _ = make_seq_batch2(self.train_data, self.batch_size)
        X_local = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_local[self.modalities[0]].shape[1]
        
        idx_end = 0
        while idx_end < seq_len:
            win_len = np.random.randint(16, 32)
            idx_start = idx_end
            idx_end = min(idx_end + win_len, seq_len)
            
            self.optimizer_total.zero_grad()
            
            # Forward
            reps_list = []
            for m in self.modalities:
                x_seg = torch.from_numpy(X_local[m][:, idx_start:idx_end, :]).float().to(self.device)
                _, out = self.ae_model.encoders[m](x_seg)
                reps_list.append(out[:, -1, :])
            
            rep_cat = torch.cat(reps_list, dim=-1)
            logits = self.cf_model(rep_cat)
            
            # 在线计算伪标签 (On-the-fly Pseudo-labeling)
            # 这种方式比预先计算更节省内存，且利用了最新梯度
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                # 再次应用阈值
                batch_mask = max_probs > self.conf_threshold
            
            if batch_mask.sum() > 0:
                # 只计算高置信度样本的 CrossEntropy
                loss = self.cls_loss_fn(logits[batch_mask], pseudo_labels[batch_mask])
                loss.backward()
                self.optimizer_total.step()
                epoch_loss.append(loss.item())
            
        return np.mean(epoch_loss) if len(epoch_loss) > 0 else 0.0