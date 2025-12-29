import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *

class UserDF(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3, beta=1.0, lamda=0.0, 
                 local_epochs=1, label_ratio=0.1):
        
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio)

        # 优化器：同时优化 AE 和 CF
        self.optimizer = torch.optim.Adam(
            list(self.ae_model.parameters()) + list(self.cf_model.parameters()),
            lr=self.learning_rate
        )
        self.rec_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def train_local(self):
        """
        FedDF 中的本地训练是标准的，不受服务端蒸馏干扰 。
        这里结合你的任务：同时训练 AE 重构和 CF 分类。
        """
        self.ae_model.train()
        self.cf_model.train()
        
        epoch_loss = []

        for ep in range(self.local_epochs):
            # 1. 准备数据
            modalities_seq, y_seq = make_seq_batch2(self.train_data, self.batch_size)
            X_local = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_local[self.modalities[0]].shape[1]
            
            # 将 y_seq 展平对齐
            # 注意：实际代码中 y 可能需要根据 seq_len 切分，这里简化处理
            y_batch_full = torch.from_numpy(y_seq).long().to(self.device)

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                self.optimizer.zero_grad()
                
                total_loss = 0.0
                reps_list = []

                # --- AE Forward & Reconstruction ---
                for m in self.modalities:
                    x_seg = torch.from_numpy(X_local[m][:, idx_start:idx_end, :]).float().to(self.device)
                    
                    # 假设 ae_model.encode 返回 (z_share, z_spec, out)
                    # 或者是 (mid, final)。根据你提供的 ae_model.py，SplitLSTMAutoEncoder forward 返回 x_recon
                    # 这里我们需要 latent 来做分类，所以调用 encode
                    
                    # 兼容 SplitLSTMAutoEncoder 逻辑
                    if hasattr(self.ae_model, 'encode'):
                         # 假设 encode 返回中间特征用于分类
                         # 注意：你需要确认 ae_model 的具体接口，这里假设它能返回特征
                         _, out = self.ae_model.encoders[m](x_seg) 
                         # 取最后时刻作为分类特征
                         rep = out[:, -1, :] 
                    else:
                        rep = torch.zeros(self.batch_size, 32).to(self.device) # Fallback

                    reps_list.append(rep)
                    
                    # 重构损失
                    x_recon = self.ae_model(x_seg, m)
                    total_loss += self.rec_loss_fn(x_recon, x_seg)

                # --- Classifier Forward ---
                # 拼接多模态特征
                if len(reps_list) > 0:
                    rep_cat = torch.cat(reps_list, dim=-1)
                    logits = self.cf_model(rep_cat)
                    
                    # 对应的标签 (取窗口内的第一个或者多数投票，这里简化取第一个)
                    # 实际数据处理需根据你的 dim 调整
                    y_seg = y_batch_full[:, idx_start].reshape(-1)
                    
                    cls_loss = self.cls_loss_fn(logits, y_seg)
                    total_loss += cls_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_loss.append(total_loss.item())

        return np.mean(epoch_loss)

    def get_logits(self, X_public_batch):
        """
        FedDF 核心辅助函数：服务端调用此函数获取该客户端在 Public Data 上的 Logits [cite: 74]。
        X_public_batch: {modality: tensor}
        """
        self.ae_model.eval()
        self.cf_model.eval()
        
        with torch.no_grad():
            reps_list = []
            for m in self.modalities:
                if m in X_public_batch:
                    x_seg = X_public_batch[m] # [B, T, D]
                    # 提取特征
                    _, out = self.ae_model.encoders[m](x_seg)
                    rep = out[:, -1, :] 
                    reps_list.append(rep)
            
            if len(reps_list) == 0:
                return None
            
            rep_cat = torch.cat(reps_list, dim=-1)
            logits = self.cf_model(rep_cat)
            return logits