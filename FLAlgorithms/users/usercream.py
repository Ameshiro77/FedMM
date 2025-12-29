import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *

class UserCream(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3, beta=1.0, lamda=0.0, 
                 local_epochs=1, label_ratio=0.1):
        
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.ae_model.parameters()) + list(self.cf_model.parameters()),
            lr=self.learning_rate
        )
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # Contrastive Hyperparameters
        self.temperature = 0.07  # InfoNCE 温度系数
        self.mu = 1.0            # 对比损失权重

    def contrastive_loss(self, z_local, z_global):
        """
        计算 InfoNCE Loss (Global-Local Contrast)
        z_local: [B, D]
        z_global: [B, D]
        """
        batch_size = z_local.shape[0]
        
        # 归一化
        z_local = F.normalize(z_local, dim=1)
        z_global = F.normalize(z_global, dim=1)
        
        # 相似度矩阵 [B, B]
        logits = torch.matmul(z_local, z_global.T) / self.temperature
        
        # 标签是对于角线 (0,0), (1,1)...
        labels = torch.arange(batch_size).to(self.device)
        
        loss = self.cls_loss_fn(logits, labels)
        return loss

    def train_contrast(self, server_reps_batch=None, public_data_batch=None):
        """
        CreamFL 本地训练：Task Loss + Contrastive Regularization
        server_reps_batch: 服务端下发的对应公共数据特征 (Global Anchors)
        public_data_batch: 对应的公共数据输入
        """
        self.ae_model.train()
        self.cf_model.train()
        
        epoch_loss = []

        for ep in range(self.local_epochs):
            # 1. 准备私有数据 (Private Data)
            modalities_seq, y_seq = make_seq_batch2(self.train_data, self.batch_size)
            X_local = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_local[self.modalities[0]].shape[1]
            y_batch_full = torch.from_numpy(y_seq).long().to(self.device)

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                self.optimizer.zero_grad()
                
                # === A. 私有数据分类损失 (Task Loss) ===
                total_loss = 0.0
                reps_list = []
                for m in self.modalities:
                    x_seg = torch.from_numpy(X_local[m][:, idx_start:idx_end, :]).float().to(self.device)
                    # 提取特征
                    # 假设 ae_model.encoders[m] 返回 (hidden, output)
                    _, out = self.ae_model.encoders[m](x_seg) 
                    rep = out[:, -1, :] 
                    reps_list.append(rep)
                
                if len(reps_list) > 0:
                    rep_cat = torch.cat(reps_list, dim=-1)
                    logits = self.cf_model(rep_cat)
                    y_seg = y_batch_full[:, idx_start].reshape(-1)
                    cls_loss = self.cls_loss_fn(logits, y_seg)
                    total_loss += cls_loss

                # === B. 公共数据对比正则化 (Contrastive Regularization) ===
                # 如果服务端下发了 Global Representations，则进行对齐
                if server_reps_batch is not None and public_data_batch is not None:
                    # 随机采样一部分公共数据与服务端对齐 (模拟 batch 对齐)
                    # 为简化实现，这里假设传入的 public_data_batch 已经是对齐好的
                    
                    # 提取本地对公共数据的特征
                    pub_reps_list = []
                    for m in self.modalities:
                        if m in public_data_batch:
                            x_pub = public_data_batch[m].to(self.device)
                            _, out_pub = self.ae_model.encoders[m](x_pub)
                            pub_reps_list.append(out_pub[:, -1, :])
                    
                    if len(pub_reps_list) > 0:
                        z_local_pub = torch.cat(pub_reps_list, dim=-1)
                        z_global_target = server_reps_batch.to(self.device)
                        
                        # CreamFL: 拉近本地公共特征与服务端公共特征
                        con_loss = self.contrastive_loss(z_local_pub, z_global_target)
                        total_loss += self.mu * con_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_loss.append(total_loss.item())

        return np.mean(epoch_loss)

    def get_public_reps(self, X_public_batch):
        """
        计算并上传公共数据特征
        """
        self.ae_model.eval()
        with torch.no_grad():
            reps_list = []
            for m in self.modalities:
                if m in X_public_batch:
                    x_seg = X_public_batch[m].to(self.device)
                    _, out = self.ae_model.encoders[m](x_seg)
                    rep = out[:, -1, :]
                    reps_list.append(rep)
            
            if len(reps_list) == 0:
                return None
            
            # 简单的特征拼接或平均，视具体实现而定
            rep_cat = torch.cat(reps_list, dim=-1)
            return rep_cat