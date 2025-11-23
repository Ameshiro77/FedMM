import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FLAlgorithms.users.userbase import User

# Implementation for clients
from utils.model_utils import *
from FLAlgorithms.config import *
from FLAlgorithms.trainmodel.losses import *


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

    def get_prototype_weight(self):
        y_all = self.labeled_data["y"]
        unique_classes, counts = np.unique(y_all, return_counts=True)
        proto_weights = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
        return proto_weights
    
    def upload_prototype(self, win_len=100):
        """
        基于[z_share, z_spec]构造时间步级原型。
        对labeled_data按固定窗口处理，每个时间步的标签参与原型聚合。
        """
        self.ae_model.eval()
        prototypes = {}  # {class_id: prototype tensor}
        feats_per_class = {}  # {class_id: [features]}
        
        X_modal = {m: self.labeled_data[m] for m in self.modalities}
        y_all = self.labeled_data["y"]
        unique_classes = np.unique(y_all)
        
        n_process = len(y_all) // win_len + 1
        for i in range(n_process):
            idx_start = i * win_len
            idx_end = min((i + 1) * win_len, len(y_all))
            if idx_end - idx_start < 2:
                continue

            # 当前窗口数据
            x_win = {m: torch.tensor(X_modal[m][idx_start:idx_end], dtype=torch.float32, device=self.device)
                    for m in self.modalities}
            y_win = torch.tensor(y_all[idx_start:idx_end], dtype=torch.long, device=self.device)

            # 多模态编码
            reps_modal = []
            for m in self.modalities:
                with torch.no_grad():
                    z_share, z_spec, _ = self.ae_model.encode(x_win[m], m)
                    # 拼接共享与特有特征 → 每个时间步 [T, D_share + D_spec]
                    z_full = torch.cat([z_share, z_spec], dim=-1)
                    reps_modal.append(z_full)

            # 拼接多模态 → [T, D_total]
            reps_cat = torch.cat(reps_modal, dim=-1)

            # 按时间步标签聚合
            for cls in unique_classes:
                mask = (y_win == cls)
                if mask.sum() == 0:
                    continue
                feats_cls = reps_cat[mask]  # [N_cls_t, D_total]
                feats_per_class.setdefault(int(cls), []).append(feats_cls)

        # 计算每类原型
        for cls, feats_list in feats_per_class.items():
            feats_all = torch.cat(feats_list, dim=0)
            proto = feats_all.mean(dim=0, keepdim=True)  # 类内平均
            prototypes[cls] = proto.detach().cpu()

        
        return prototypes

    def upload_specs(self):
        self.ae_model.eval()
        z_specs_dict = {m: None for m in self.modalities}

        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_modal[self.modalities[0]].shape[1]  # 64 100 63

        idx_end = 0
        win_num = 0
        temp_z_windows = {m: [] for m in self.modalities}  # 用于存窗口
        
        for m in self.modalities:
            x_m = torch.from_numpy(X_modal[m]).to(self.device)  
            z_share, z_spec, _ = self.ae_model.encode(x_m,m)       
            z_mean = z_spec.mean(dim=1)                         # [B, D]
            z_specs_dict[m] = z_mean.detach().cpu()
  
        return z_specs_dict

    def upload_shares(self):
        self.ae_model.eval()
        z_shares_dict = {m: None for m in self.modalities}

        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_modal[self.modalities[0]].shape[1]  # 64 100 63

        idx_end = 0
        win_num = 0
        temp_z_windows = {m: [] for m in self.modalities}  # 用于存窗口
        
        for m in self.modalities:
            x_m = torch.from_numpy(X_modal[m]).to(self.device)  
            z_share, z_spec, _ = self.ae_model.encode(x_m,m)       
            z_mean = z_share.mean(dim=1)                         # [B, D]
            z_shares_dict[m] = z_mean.detach().cpu()
  
        return z_shares_dict

    def train_ae_prop(self, global_share, global_prototypes, pre_w):
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
                rec_loss, orth_loss, align_loss, reg_loss ,proto_loss = 0.0, 0.0, 0.0, 0.0,0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    x_recon, z_share, z_spec = self.ae_model(x_m)

                    
                    # ✅ 按模态记录
                    z_shares_all[m].append(z_share)

                    # 1. 本地重构 + 正交损失
                    rec_loss += self.rec_loss_fn(x_recon, x_m)
                    # orth_term = torch.mean(torch.abs(
                    #     torch.sum(F.normalize(z_share, dim=-1) * F.normalize(z_spec, dim=-1), dim=-1)
                    # ))
                    # orth_loss += orth_term
                    orth_loss += hsic_loss(z_share, z_spec)

                    # # 2. 对齐全局共享向量
                    # if global_share is not None:
                    #     z_global_srv = global_share.detach().to(self.device)
                    #     # print(z_share.shape,z_global_srv.shape)
                    #     align_loss_term = torch.mean(torch.sum((z_share - z_global_srv.unsqueeze(0))**2, dim=1))
                    #     align_loss += align_loss_term

                    # 2. 对齐全局共享向量（使用余弦相似度）
                    if global_share is not None:
                        z_global_srv = global_share.detach().to(self.device)
                        # z_global_expand = z_global_srv.unsqueeze(0).expand_as(z_share)
                        # z_share: [B, T, D]
                        # z_global_srv: [B, D] -> 扩展成 [B, T, D]
                        print(z_global_srv.shape,z_share.shape)
            

                        cos_sim = F.cosine_similarity(z_share, z_global_srv, dim=1)
                        align_loss_term = 1 - cos_sim.mean()
                        align_loss += align_loss_term


                    # 3. 上一轮参数正则
                    if pre_w is not None:
                        reg = 0.0
                        for p_new, p_old in zip(self.ae_model.parameters(), pre_w):
                            reg += torch.sum((p_new - p_old.to(self.device))**2)
                        reg_loss += reg
                    
                    # # 4.
                    # z_share_2, z_spec_2, _ = self.ae_model.encode(x_m, m)
                    # z_full = torch.cat([z_share_2, z_spec_2], dim=-1)
                    # if global_prototypes is not None and hasattr(self, "labeled_data"):
                    #     y_true = torch.from_numpy(self.labeled_data["y"]).to(self.device)  # [B, T]
                    #     print(z_full.shape,y_true.shape)
                    #     # flatten：与时间步级标签对应
                    #     B, T, D = z_full.shape
                    #     z_flat = z_full.reshape(B * T, D)
                    #     y_flat = y_true.reshape(B * T)

                    #     for cls_id, proto_g in global_prototypes.items():
                    #         mask = (y_flat == cls_id)
                    #         if mask.sum() > 0:
                    #             z_cls = z_flat[mask]
                    #             proto_loss += F.mse_loss(z_cls.mean(dim=0), proto_g.to(self.device))


                # === 总损失 ===
                loss = (
                    rec_loss
                    + 0.1 * orth_loss
                    + 1.0 * align_loss
                    + 0.1 * reg_loss
                )

                # 汇总损失
                # loss = rec_loss + 0.01 * orth_loss + 0.1 * align_loss + 0.01 * reg_loss
                loss = rec_loss + 0.1 * orth_loss + 0.5 * align_loss
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

    def train_cf_prop(self, server_logits_bank=None, alpha=0.5, T=3.0):

        self.freeze(self.ae_model)  # AE 不更新
        self.cf_model.train()

        # 初始化损失字典
        loss_dict_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
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
                    # diff from previous
                    z_share, z_spec, out = self.ae_model.encode(x_m, m)
                    z_latent = torch.cat([z_share, z_spec], dim=-1)
                    latents.append(z_latent)

                # 拼接模态特征
                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()

                logits = self.cf_model(latent_cat)

                # === 1. 基础分类损失 ===
                ce_loss = self.cls_loss_fn(logits, y_batch)

                # === 2. 蒸馏损失（若服务端下发了 logits） ===
                kd_loss = 0.0
                if server_logits_bank is not None:
                    # 从 bank 中取出教师 logits
                    teacher_logits = []
                    for y in y_batch.detach().cpu().numpy():
                        if y in server_logits_bank:
                            teacher_logits.append(server_logits_bank[y].unsqueeze(0))
                        else:
                            # 若没有该类，则用全零向量
                            teacher_logits.append(torch.zeros_like(logits[0:1]))

                    teacher_logits = torch.cat(teacher_logits, dim=0).to(self.device)

                    # KL散度蒸馏损失
                    p_s = F.log_softmax(logits / T, dim=1)
                    p_t = F.softmax(teacher_logits / T, dim=1)
                    kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

                # === 3. 合并损失 ===
                loss = (1 - alpha) * ce_loss + alpha * kd_loss

                # # print(logits.shape, y_batch.shape)
                # loss = self.cls_loss_fn(logits, y_batch)


                loss.backward()
                self.optimizer_cf.step()

                # 计算准确率
                pred = torch.argmax(logits, dim=-1)  # [B, W]
                correct = (pred == y_batch).sum().item()
                total_samples += y_batch.numel()

                
                # 记录当前batch的损失和准确率
                batch_loss = loss.item()
                batch_acc = correct / y_batch.size(0)
                
                loss_dict_epoch['cf_loss'].append(batch_loss)
                loss_dict_epoch['cf_acc'].append(batch_acc)

                total_loss += batch_loss
                total_correct += correct
                total_samples += y_batch.size(0)

            # 打印每个epoch的训练信息
            epoch_avg_loss = np.mean(loss_dict_epoch['cf_loss'])
            epoch_avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            print(
                f"[Client {self.client_id}] CF Epoch {ep+1}/{self.local_epochs} | "
                f"Loss={epoch_avg_loss:.4f} | Acc={epoch_avg_acc:.4f}"
            )

        self.unfreeze(self.ae_model)
        
        # 计算平均损失和准确率
        loss_dict_avg = {
            'cf_loss': np.mean(loss_dict_epoch['cf_loss']),
            'cf_acc': np.mean(loss_dict_epoch['cf_acc'])
        }
        
        return loss_dict_avg

    def test(self, eval_win = EVAL_WIN):
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
                    # print(batch_x[m].shape, m)
                    z_share, z_spec, out = self.ae_model.encode(batch_x[m], m)
                    z_latent = torch.cat([z_share, z_spec], dim=-1)
                    # print(z_share.shape, z_spec.shape, z_latent.shape)
                    reps.append(z_latent)

                # 拼接多个模态特征
                reps = torch.cat(reps, dim=-1)

                # ------- classifier -------
                outputs = self.cf_model(reps)
                # print(outputs.shape, batch_y.shape)
                loss = self.cls_loss_fn(outputs, batch_y)

                # ------- 统计 -------
                total_loss += loss.item() * batch_y.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_acc, avg_loss

    def save_model(self, save_dir="./saved_clients"):
        """
        保存该客户端的模型（AE + CF）及客户端标识。
        保存格式：
            save_dir/client_{id}_ae.pt
            save_dir/client_{id}_cf.pt
        """
        os.makedirs(save_dir, exist_ok=True)

        ae_path = os.path.join(save_dir, f"client_{self.client_id}_ae.pt")
        cf_path = os.path.join(save_dir, f"client_{self.client_id}_cf.pt")

        # 只保存 state_dict，不保存整个对象，兼容性最好
        torch.save({
            "client_id": self.client_id,
            "ae_state": self.ae_model.state_dict()
        }, ae_path)

        torch.save({
            "client_id": self.client_id,
            "cf_state": self.cf_model.state_dict()
        }, cf_path)

        print(f"[Client {self.client_id}] 模型已保存：")
        print(f" - AE: {ae_path}")
        print(f" - CF: {cf_path}")
        
    def load_model(self, ae_path, cf_path):
        """加载 AE 与 CF 模型权重"""
        ae_ckpt = torch.load(ae_path, map_location=self.device)
        cf_ckpt = torch.load(cf_path, map_location=self.device)

        self.ae_model.load_state_dict(ae_ckpt["ae_state"])
        self.cf_model.load_state_dict(cf_ckpt["cf_state"])

        print(f"[Client {self.client_id}] 模型加载完毕")
