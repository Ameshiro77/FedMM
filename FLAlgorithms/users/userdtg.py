import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *

class UserDTG(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)

    def upload_spec(self):
        """
        上传本地特征的统计量 (Mean, Std) 给服务端生成器使用
        """
        self.ae_model.eval()
        z_specs_stat = {m: {'mean': None, 'std': None} for m in self.modalities}

 
        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        
        with torch.no_grad():
            for m in self.modalities:
                x_m = torch.from_numpy(X_modal[m]).to(self.device)
                

                _, z_lat = self.ae_model.encode(x_m, m) 
    
                z_flat = z_lat.reshape(-1, z_lat.shape[-1])
                
                spec_mean = z_flat.mean(dim=0)
                spec_std = z_flat.std(dim=0) + 1e-6 # 防止除0

                z_specs_stat[m]['mean'] = spec_mean.detach().cpu()
                z_specs_stat[m]['std'] = spec_std.detach().cpu()
  
        return z_specs_stat

    def train_ae(self):
        """
        标准 AE 训练 (参考 UserFedAvg)
        """
        self.ae_model.train()
        self.cf_model.eval()
        rec_loss_lst = []
        
        for i in range(self.local_epochs):
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
                    # 标准 AE forward
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)

                rec_loss.backward()
                self.optimizer_ae.step()
                rec_loss_lst.append(rec_loss.item())

        return np.mean(rec_loss_lst) if len(rec_loss_lst) > 0 else 0.0


        return np.mean(loss_lst), np.mean(acc_lst)