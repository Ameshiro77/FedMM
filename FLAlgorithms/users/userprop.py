import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FLAlgorithms.users.userbase import User

# Implementation for clients
from utils.model_utils import *


class UserProp(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)

    # TODO
    # 记得baseline改成train data

    def upload_shares(self):
        self.ae_model.eval()
        z_shares_dict = {m: None for m in self.modalities}

        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_modal[self.modalities[0]].shape[1]  # 64 100 63

        idx_end = 0
        win_num = 0
        temp_z_windows = {m: [] for m in self.modalities}  # 用于存窗口

        while idx_end < seq_len:
            win_num += 1
            win_len = np.random.randint(16, 32)
            idx_start = idx_end
            idx_end = min(idx_end + win_len, seq_len)

            for m in self.modalities:
                x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                z_share, z_spec, _ = self.ae_model.encode(x_m)  # [batch, D]
                temp_z_windows[m].append(z_share.detach().cpu())

        # 对每个模态的所有窗口做平均
        for m in self.modalities:
            z_all_windows = torch.cat(temp_z_windows[m], dim=0)  # [batch*num_windows, D]
            z_mean = z_all_windows.mean(dim=0)  # [D]
            z_shares_dict[m] = z_mean
 
        return z_shares_dict


    def train_ae_prop(self, global_share, pre_w):
        """
        训练自编码器
        - global_share: 全局共享特征向量，用于对齐
        - pre_w: 上一轮 encoder 参数，用于正则化
        """
        self.ae_model.train()
        self.cf_model.eval()
        rec_loss_lst = []

        # 改：用 dict 存各模态的 z_share
        z_shares_all = {m: [] for m in self.modalities}

        loss_dict_epoch = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}

        for ep in range(self.local_epochs):
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
                rec_loss, orth_loss, align_loss, reg_loss = 0.0, 0.0, 0.0, 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    x_recon, z_share, z_spec = self.ae_model(x_m)

                    # ✅ 按模态记录
                    z_shares_all[m].append(z_share)

                    # 1. 本地重构 + 正交损失
                    rec_loss += self.rec_loss_fn(x_recon, x_m)
                    orth_term = torch.mean(torch.abs(
                        torch.sum(F.normalize(z_share, dim=-1) * F.normalize(z_spec, dim=-1), dim=-1)
                    ))
                    orth_loss += orth_term

                    # 2. 对齐全局共享向量
                    if global_share is not None:
                        z_global_srv = global_share.detach().to(self.device)
                        # print(z_share.shape,z_global_srv.shape)
                        align_loss_term = torch.mean(torch.sum((z_share - z_global_srv.unsqueeze(0))**2, dim=1))
                        align_loss += align_loss_term

                    # 3. 上一轮参数正则
                    if pre_w is not None:
                        reg = 0.0
                        for p_new, p_old in zip(self.ae_model.parameters(), pre_w):
                            reg += torch.sum((p_new - p_old.to(self.device))**2)
                        reg_loss += reg

                # 汇总损失
                loss = rec_loss + 0.01 * orth_loss + 0.1 * align_loss
                loss.backward()
                self.optimizer_ae.step()

                # 记录
                loss_dict_epoch['rec_loss'].append(rec_loss.item())
                loss_dict_epoch['orth_loss'].append(orth_loss.item())
                loss_dict_epoch['align_loss'].append(align_loss.item())
                loss_dict_epoch['reg_loss'].append(reg_loss.item())

                total_loss += loss.item()
                rec_loss_lst.append(rec_loss.item())

            print(
                f"[Client {self.client_id}] Epoch {ep+1}/{self.local_epochs} | "
                f"Recon={np.mean(loss_dict_epoch['rec_loss']):.4f} | "
                f"Orth={np.mean(loss_dict_epoch['orth_loss']):.4f} | "
                f"Align={np.mean(loss_dict_epoch['align_loss']):.4f} | "
                f"Reg={np.mean(loss_dict_epoch['reg_loss']):.4f}"
            )

        # ✅ 每个模态的 z_share 拼接成一个整体张量
        z_shares_all = {m: torch.cat(v, dim=0) for m, v in z_shares_all.items()}

        # ✅ 平均损失
        loss_dict_avg = {k: np.mean(v) for k, v in loss_dict_epoch.items()}

        # ✅ 返回“模态-特征字典”和“损失字典”
        return z_shares_all, loss_dict_avg


    def train_cf_prop(self):
        self.freeze(self.ae_model)  # AE 不更新
        self.cf_model.train()

        total_loss, total_correct, total_samples = 0.0, 0, 0

        # 构造有监督数据序列
        modalities_seq, y_seq = make_seq_batch2(self.labeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_modal[self.modalities[0]].shape[1]

        idx_end = 0
        while idx_end < seq_len:
            win_len = np.random.randint(16, 32)  # 随机窗口长度
            idx_start = idx_end
            idx_end = min(idx_end + win_len, seq_len)

            self.optimizer_cf.zero_grad()
            latents = []

            # 每个模态编码
            for m in self.modalities:
                x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                _, z_spec, out = self.ae_model.encode(x_m, m)
                latents.append(z_spec)

            # 拼接模态特征
            latent_cat = torch.cat(latents, dim=1)
            y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).long()

            logits = self.cf_model(latent_cat)
            loss = self.cls_loss_fn(logits, y_batch[:, 0])

            loss.backward()
            self.optimizer_cf.step()

            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=1) == y_batch[:, 0]).sum().item()
            total_samples += y_batch.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.unfreeze(self.ae_model)
        return total_loss / max(1, seq_len // self.batch_size), acc

    def test(self, eval_win=2000):
        """测试客户端模型性能，不依赖具体 AE 结构"""
        self.ae_model.eval()
        self.cf_model.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            # 滑窗遍历测试集
            for start in range(0, len(self.test_data["y"]) - eval_win + 1, eval_win):
                # 构造多模态输入
                batch_x = {
                    m: torch.tensor(
                        self.test_data[m][start:start + eval_win],
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    for m in self.modalities
                }
                batch_y = torch.tensor(
                    self.test_data["y"][start:start + eval_win],
                    dtype=torch.long, device=self.device
                )

                # ------- encode -------
                reps = []
                for m in self.modalities:
                    # encode() 返回 (z_share, z_spec)
                    _, z_spec, out = self.ae_model.encode(batch_x[m], m)

                    reps.append(out)

                # 拼接多个模态特征
                reps = torch.cat(reps, dim=-1)

                # ------- classifier -------
                outputs = self.cf_model(reps)
                loss = self.cls_loss_fn(outputs, batch_y)

                # ------- 统计 -------
                total_loss += loss.item() * batch_y.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_acc, avg_loss
