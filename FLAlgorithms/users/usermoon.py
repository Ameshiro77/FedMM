import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FLAlgorithms.users.userbase import User
from utils.model_utils import make_seq_batch2

class UserMoon(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)
        
        # FedProx 系数（用于 L2 prox 正则）
        self.mu_prox = 0.01

        # MOON 超参数
        self.mu_moon = 0.1      # contrastive loss 权重（可调）
        self.temperature = 0.5  # temperature tau
        self.cos = nn.CosineSimilarity(dim=1).to(self.device)

    def train_ae(self, epochs=1, global_model=None, prev_model=None):
        """
        训练自编码器（AE）部分，支持：
        - FedProx（prox_loss, 使用 self.mu_prox）
        - MOON 风格的表示对比（如果传入 global_model 和 prev_model，则计算）
        参数:
            global_model: 服务器下发的全局 ae_model（用于正样本）
            prev_model:    本客户端上一轮保存的本地模型（用于负样本）
        返回:
            mean reconstruction loss (float)
        """
        self.ae_model.train()
        rec_loss_lst = []

        # 处理 global_model / prev_model（均为 model 对象）
        gen_model = None
        prev_local = None
        if global_model is not None:
            gen_model = copy.deepcopy(global_model).to(self.device)
            gen_model.eval()
            for p in gen_model.parameters():
                p.requires_grad = False

        if prev_model is not None:
            prev_local = copy.deepcopy(prev_model).to(self.device)
            prev_local.eval()
            for p in prev_local.parameters():
                p.requires_grad = False

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
                prox_loss = 0.0
                moon_loss = 0.0

                # AE reconstruction for each modality
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)

                # FedProx prox term (L2 between local and global)
                if gen_model is not None:
                    for local_param, global_param in zip(self.ae_model.parameters(), gen_model.parameters()):
                        prox_loss += torch.norm(local_param - global_param, p=2) ** 2

                # MOON 对比损失：对每个模态计算表示并做 InfoNCE
                # 假设 ae_model.encode(x, m) 返回 (something, rep_sequence)，我们取最后时刻向量 rep_seq[:, -1, :]
                if gen_model is not None and prev_local is not None:
                    for m in self.modalities:
                        x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                        # 取本地当前表示
                        _, rep_local_seq = self.ae_model.encode(x_m, m)
                        z_local = rep_local_seq[:, -1, :]  # (batch, dim)
                        # 全局表示
                        _, rep_gen_seq = gen_model.encode(x_m, m)
                        z_gen = rep_gen_seq[:, -1, :]
                        # 上一轮本地表示
                        _, rep_prev_seq = prev_local.encode(x_m, m)
                        z_prev = rep_prev_seq[:, -1, :]

                        # 余弦相似度
                        pos = self.cos(z_local, z_gen) / self.temperature  # (batch,)
                        neg = self.cos(z_local, z_prev) / self.temperature

                        # InfoNCE-like 二元形式
                        # loss = -log( exp(pos) / (exp(pos) + exp(neg)) )
                        # 对 batch 求均值
                        exp_pos = torch.exp(pos)
                        exp_neg = torch.exp(neg)
                        denom = exp_pos + exp_neg + 1e-8
                        sample_loss = -torch.log(exp_pos / denom + 1e-8)
                        moon_loss += sample_loss.mean()

                total_loss_value = rec_loss
                if gen_model is not None:
                    total_loss_value = total_loss_value + (self.mu_prox / 2) * prox_loss
                if (gen_model is not None) and (prev_local is not None):
                    total_loss_value = total_loss_value + self.mu_moon * moon_loss

                total_loss_value.backward()
                self.optimizer_ae.step()

                rec_loss_lst.append(rec_loss.item())

        return np.mean(rec_loss_lst) if len(rec_loss_lst) > 0 else 0.0

    def train_all(self, alpha=0.1, beta=0.1, global_model=None, prev_model=None):
        """
        联合训练 AE + 分类器（或 downstream 模块），同时添加 MOON 风格对比到分类阶段。
        参数:
            global_model: 用于 MOON 的全局模型（通常为 ae_model 的全局副本）
            prev_model:   该客户端上一轮本地模型（用于 negative）
        返回:
            total_loss, rec_loss, cls_loss, acc
        """
        self.ae_model.train()
        self.cf_model.train()

        total_loss, total_rec, total_cls = 0.0, 0.0, 0.0
        total_correct, total_samples = 0, 0

        modalities_seq, y_seq = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        y_modal = y_seq
        seq_len = X_modal[self.modalities[0]].shape[1]

        # 预处理全局与 prev（同 train_ae）
        gen_model = None
        prev_local = None
        if global_model is not None:
            gen_model = copy.deepcopy(global_model).to(self.device)
            gen_model.eval()
            for p in gen_model.parameters():
                p.requires_grad = False
        if prev_model is not None:
            prev_local = copy.deepcopy(prev_model).to(self.device)
            prev_local.eval()
            for p in prev_local.parameters():
                p.requires_grad = False

        idx_end = 0
        for epoch in range(self.local_epochs):
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer.zero_grad()
                rec_loss, cls_loss = 0.0, 0.0
                latents = []

                # AE 重构 & 收集表示
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)
                    # encode 返回 (.., rep_seq)
                    _, rep_seq = self.ae_model.encode(x_m, m)
                    latents.append(rep_seq[:, -1, :])

                # 分类
                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_modal[:, idx_start:idx_end]).to(self.device).long()
                logits = self.cf_model(latent_cat)
                cls_loss = self.cls_loss_fn(logits, y_batch[:, 0])

                # MOON 对比损失：这里用联合表示 latent_cat 做对比（如果提供 global & prev）
                moon_loss = 0.0
                if gen_model is not None and prev_local is not None:
                    # 需要拿到 global / prev 在同样输入上的表示
                    # 先按模态单独encode，再拼接（与上面相同的顺序）
                    gen_latents = []
                    prev_latents = []
                    for m in self.modalities:
                        x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                        _, rep_gen_seq = gen_model.encode(x_m, m)
                        _, rep_prev_seq = prev_local.encode(x_m, m)
                        gen_latents.append(rep_gen_seq[:, -1, :])
                        prev_latents.append(rep_prev_seq[:, -1, :])
                    z_gen = torch.cat(gen_latents, dim=1)   # (batch, dim_cat)
                    z_prev = torch.cat(prev_latents, dim=1)
                    z_local = latent_cat  # (batch, dim_cat)

                    pos = self.cos(z_local, z_gen) / self.temperature
                    neg = self.cos(z_local, z_prev) / self.temperature

                    exp_pos = torch.exp(pos)
                    exp_neg = torch.exp(neg)
                    denom = exp_pos + exp_neg + 1e-8
                    sample_loss = -torch.log(exp_pos / denom + 1e-8)
                    moon_loss = sample_loss.mean()

                loss = self.beta * rec_loss + cls_loss
                # 若有 prox term（针对 ae_model），加入 prox
                prox_loss = 0.0
                if gen_model is not None:
                    for local_param, global_param in zip(self.ae_model.parameters(), gen_model.parameters()):
                        prox_loss += torch.norm(local_param - global_param, p=2) ** 2
                    loss = loss + (self.mu_prox / 2) * prox_loss

                if moon_loss != 0.0:
                    loss = loss + self.mu_moon * moon_loss

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
