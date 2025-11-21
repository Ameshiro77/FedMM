import copy
import numpy as np
import torch
import torch.nn as nn
from utils.model_utils import *
from FLAlgorithms.users.userbase import User
from utils.train_utils import *


class UserMEKT(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3, beta=1.0, lamda=0.0, 
                 local_epochs=1, label_ratio=0.1):
        
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate, beta, lamda, local_epochs, label_ratio)
        
        # loss 函数
        self.rec_loss_fn = nn.MSELoss().to(self.device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.temperature = 0.5

        # MEKT 特有超参数
        self.alpha = 0.5  # 蒸馏损失权重
        self.beta = 0.5   # 对比损失权重


        self.optimizer = torch.optim.Adam(
            list(self.ae_model.parameters()) + list(self.cf_model.parameters()),
            lr=self.learning_rate
        )

    def freeze(self, sub_model):
        for p in sub_model.parameters():
            p.requires_grad = False

    def unfreeze(self, sub_model):
        for p in sub_model.parameters():
            p.requires_grad = True


    def train_ae_distill(self, epochs, global_ae):
        gen_model = copy.deepcopy(global_ae).to(self.device)
        gen_model.eval()
        for p in gen_model.parameters():
            p.requires_grad = False

        self.ae_model.train()
        Rec_epoch_losses, KD_epoch_losses = [], []

        for ep in range(epochs):
            batch_size = self.batch_size

            # === 读取本地训练数据 ===
            modalities_seq, _ = make_seq_batch2(self.train_data, batch_size)
            X_local = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_local[self.modalities[0]].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer.zero_grad()
                rec_loss, kd_loss = 0.0, 0.0

                # === Step 1: 生成本客户端的 ESH ===
                public_seq, _ = make_seq_batch2(self.public_data, batch_size)
                X_pub = {m: public_seq[m] for m in self.modalities}
                ESH_list = []

                for modal in self.modalities:
                    x_pub = X_pub[modal][:, idx_start:idx_end, :]
                    seq_pub = torch.from_numpy(x_pub).float().to(self.device)
                    _, rep_pub_global = gen_model.encode(seq_pub, modal)
                    ESH_list.append(rep_pub_global.detach())

                # 拼接当前客户端的多模态特征
                ESH_local = torch.cat(ESH_list, dim=-1)

                # === Step 2: 本地自编码器蒸馏训练 ===
                for modal in self.modalities:
                    x_local = X_local[modal][:, idx_start:idx_end, :]
                    seq_local = torch.from_numpy(x_local).float().to(self.device)

                    x_pub = X_pub[modal][:, idx_start:idx_end, :]
                    seq_pub = torch.from_numpy(x_pub).float().to(self.device)

                    rec_out = self.ae_model(seq_local, modal)
                    _, rep_pub_local = self.ae_model.encode(seq_pub, modal)

                    rec_loss += self.rec_loss_fn(rec_out, seq_local)
                    kd_loss += self.kl_loss_fn(
                        F.log_softmax(rep_pub_local / self.temperature, dim=-1),
                        F.softmax(ESH_local.detach() / self.temperature, dim=-1)
                    ) * (self.temperature ** 2)

                loss = rec_loss + self.alpha * kd_loss
                loss.backward()
                self.optimizer.step()

                Rec_epoch_losses.append(rec_loss.item())
                KD_epoch_losses.append(kd_loss.item())

        return np.mean(Rec_epoch_losses), np.mean(KD_epoch_losses)


