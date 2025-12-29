import copy

import numpy as np
import torch
import torch.nn as nn


from FLAlgorithms.users.userbase import User

# Implementation for clients
from utils.model_utils import *


class UserFedprox(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)
        
        self.mu = 3.0


    def train_ae_prox(self, global_model=None):
        self.ae_model.train()
        self.cf_model.eval()
        rec_loss_lst = []
        
        # FedProx特有：处理全局模型
        if global_model is not None:
            global_model = copy.deepcopy(global_model)
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
        
        for i in range(self.local_epochs):
            total_loss = 0.0
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
                prox_loss = 0.0

                # 重建损失
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)

                # FedProx正则项 - 只计算客户端拥有的模态
                if global_model is not None:
                    local_state_dict = self.ae_model.state_dict()
                    global_state_dict = global_model.state_dict()
                    
                    for modality in self.modalities:
                        # 只计算该客户端拥有的模态对应的参数
                        for key in local_state_dict:
                            if f"encoders.{modality}." in key or f"decoders.{modality}." in key:
                                if key in global_state_dict and local_state_dict[key].shape == global_state_dict[key].shape:
                                    local_param = local_state_dict[key]
                                    global_param = global_state_dict[key]
                                    prox_loss += torch.norm(local_param - global_param, p=2) ** 2
                
                total_loss_value = rec_loss + (self.mu / 2) * prox_loss
                print("prox loss:", prox_loss)
                total_loss_value.backward()
                self.optimizer_ae.step()
                total_loss += total_loss_value.item()
    
    def train_all(self, alpha=0.1, beta=0.1):
        self.ae_model.train()
        self.cf_model.train()
        total_loss, total_rec, total_cls = 0.0, 0.0, 0.0
        total_correct, total_samples = 0, 0

        # 将训练数据切成 batch first 的 seq
        # modalities_seq, y_seq = make_seq_batch(self.train_data, self.idxs, self.seg_len, self.batch_size)
        modalities_seq, y_seq = make_seq_batch2(self.unlabeled_data,self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        y_modal = y_seq

        seq_len = X_modal[self.modalities[0]].shape[1]

        idx_end = 0
        for epoch in range(self.local_epochs):
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer.zero_grad()
                rec_loss, cls_loss = 0.0, 0.0
                latents = []

                # AE 重构
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)
                    latents.append(self.ae_model.encode(x_m, m)[1][:, -1, :])  

                # 分类
                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_modal[:, idx_start:idx_end]).to(self.device).long()
                logits = self.cf_model(latent_cat)
                cls_loss = self.cls_loss_fn(logits, y_batch[:, 0])  

                loss = self.beta * rec_loss + cls_loss
                loss.backward()
                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                total_rec += rec_loss.item()
                total_cls += cls_loss.item()
                total_correct += (torch.argmax(logits, dim=1) == y_batch[:, 0]).sum().item()
                total_samples += y_batch.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Client id: {self.client_id}, Train loss: {total_loss:.4f}, Train acc: {acc:.4f}")
        return total_loss, total_rec, total_cls, acc
