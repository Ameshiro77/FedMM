import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *
from FLAlgorithms.config import *
from FLAlgorithms.trainmodel.losses import *


class UserProp(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1, args=None):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs,
                         label_ratio)

        # === 配置参数 ===
        self.conf_threshold = 0.9  # 伪标签置信度阈值 (用于细粒度对齐)
        # self.orth_weight = 0.1     # 固定正交损失权重 (不再动态变化)

        self.client_orth_weight = args.client_orth_weight
        self.client_align_weight = args.client_align_weight
        self.client_reg_weight = args.client_reg_weight
        self.client_logits_weight = args.client_logits_weight
        
        # self.client_dyn_alpha = args.client_dyn_alpha

    def get_prototype_weight(self):
        y_all = self.labeled_data["y"]
        unique_classes, counts = np.unique(y_all, return_counts=True)
        proto_weights = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
        return proto_weights

    def upload_prototype(self, win_len=100):
        self.ae_model.eval()
        prototypes = {}
        feats_per_class = {}

        X_modal = {m: self.labeled_data[m] for m in self.modalities}
        y_all = self.labeled_data["y"]
        unique_classes = np.unique(y_all)

        n_process = len(y_all) // win_len + 1
        for i in range(n_process):
            idx_start = i * win_len
            idx_end = min((i + 1) * win_len, len(y_all))
            if idx_end - idx_start < 2:
                continue

            x_win = {m: torch.tensor(X_modal[m][idx_start:idx_end], dtype=torch.float32, device=self.device)
                     for m in self.modalities}
            y_win = torch.tensor(y_all[idx_start:idx_end], dtype=torch.long, device=self.device)

            reps_modal = []
            for m in self.modalities:
                with torch.no_grad():
                    z_share, z_spec, _ = self.ae_model.encode(x_win[m], m)
                    z_full = torch.cat([z_share, z_spec], dim=-1)
                    reps_modal.append(z_full)

            reps_cat = torch.cat(reps_modal, dim=-1)

            for cls in unique_classes:
                mask = (y_win == cls)
                if mask.sum() == 0:
                    continue
                feats_cls = reps_cat[mask]
                feats_per_class.setdefault(int(cls), []).append(feats_cls)

        for cls, feats_list in feats_per_class.items():
            feats_all = torch.cat(feats_list, dim=0)
            proto = feats_all.mean(dim=0, keepdim=True)
            prototypes[cls] = proto.detach().cpu()

        return prototypes

    def upload_shares(self):
        self.ae_model.eval()
        z_shares_dict = {m: None for m in self.modalities}

        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}

        for m in self.modalities:
            x_m = torch.from_numpy(X_modal[m]).to(self.device)
            z_share, z_spec, _ = self.ae_model.encode(x_m, m)
            z_mean = z_share.mean(dim=1)
            z_shares_dict[m] = z_mean.detach().cpu()

        return z_shares_dict

    def train_ae_prop(self, global_share, global_prototypes, pre_w):
        """
        [优化版] 自编码器训练
        1. 伪标签细粒度对齐 (Fine-grained Alignment)
        2. 无 Warm-up，直接使用固定 orth_weight
        """
        self.ae_model.train()
        self.cf_model.eval()  # 分类器只做预测，不更新

        z_shares_all = {m: [] for m in self.modalities}
        loss_dict_epoch = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}

        # === 预处理全局原型 ===
        proto_tensor = None
        if global_prototypes is not None:
            max_cls = max(global_prototypes.keys())
            sample_val = list(global_prototypes.values())[0]
            rep_dim = sample_val.shape[-1]

            proto_tensor = torch.zeros(max_cls + 1, rep_dim).to(self.device)
            for k, v in global_prototypes.items():
                proto_tensor[k] = v.to(self.device)

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
                rec_loss, orth_loss, align_loss, reg_loss = 0.0, 0.0, 0.0, 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    x_recon, z_share, z_spec = self.ae_model(x_m)

                    z_shares_all[m].append(z_share.detach().cpu())

                    # 1. 重构损失
                    rec_loss += self.rec_loss_fn(x_recon, x_m)

                    # 2. 正交损失 (固定权重)
                    current_orth = hsic_loss(z_share, z_spec)

                    # orth_term = torch.mean(torch.abs(
                    #     torch.sum(F.normalize(z_share, dim=-1) * F.normalize(z_spec, dim=-1), dim=-1)
                    # ))
                    # orth_loss += orth_term

                    orth_loss += current_orth

                    # if global_share is not None:
                    #     z_global_srv = global_share.detach().to(self.device)
                    #     # print(z_global_srv.shape,z_share.shape)
                    #     cos_sim = F.cosine_similarity(z_share, z_global_srv, dim=1)
                    #     align_loss_term = 1 - cos_sim.mean()
                    #     align_loss += align_loss_term

                    # =========================================================
                    # === 3. 对齐损失 (Alignment Loss) - 新旧逻辑兼容 ===
                    # =========================================================
                    if global_share is not None:
                        # -----------------------------------------------------
                        # 【新逻辑】如果传入的是字典，说明是 Modality-Specific Anchors
                        # -----------------------------------------------------
                        if isinstance(global_share, dict):
                            # 只对齐当前模态 m 对应的锚点
                            if m in global_share:
                                # 取出该模态的全局锚点 [D]
                                z_anchor = global_share[m].to(self.device)
                                
                                # 处理维度匹配：z_share 可能是 [B, T, D] 或 [B, D]
                                # 为了计算 cosine similarity，我们需要统一维度 (取时间均值)
                                if z_share.dim() == 3:
                                    z_local = z_share.mean(dim=1)
                                else:
                                    z_local = z_share
                                
                                # 计算余弦相似度 (在特征维度 dim=-1)
                                # z_local: [B, D], z_anchor: [D] (会自动广播)
                                cos_sim = F.cosine_similarity(z_local, z_anchor, dim=-1)
                                align_loss += (1 - cos_sim.mean())
                        
                        # -----------------------------------------------------
                        # 【旧逻辑】如果传入的是 Tensor，说明是 Unified z_fuse
                        # -----------------------------------------------------
                        else:
                            z_global_srv = global_share.detach().to(self.device)
                            # 原有逻辑：直接计算相似度 (保留你原始代码的 dim=1 设置)
                            cos_sim = F.cosine_similarity(z_share, z_global_srv, dim=1)
                            align_loss_term = 1 - cos_sim.mean()
                            align_loss += align_loss_term

                    # 5. 正则化
                    if pre_w is not None:
                        reg = 0.0
                        for p_new, p_old in zip(self.ae_model.parameters(), pre_w):
                            reg += torch.sum((p_new - p_old.to(self.device))**2)
                        reg_loss += reg

                # === 总损失  ===
                # loss = rec + orth + align + reg
                # loss = rec_loss + self.orth_weight * orth_loss + 1.0 * align_loss + 0.01 * reg_loss
                loss = rec_loss + \
                    self.client_orth_weight * orth_loss + \
                    self.client_align_weight * align_loss + \
                    self.client_reg_weight * reg_loss

                loss.backward()
                self.optimizer_ae.step()

                loss_dict_epoch['rec_loss'].append(rec_loss.item())
                loss_dict_epoch['orth_loss'].append(orth_loss.item()
                                                    if isinstance(orth_loss, torch.Tensor) else orth_loss)
                loss_dict_epoch['align_loss'].append(align_loss.item())
                loss_dict_epoch['reg_loss'].append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)

        z_shares_all = {m: torch.cat(v, dim=0) for m, v in z_shares_all.items()}
        loss_dict_avg = {k: np.mean(v) for k, v in loss_dict_epoch.items()}
        return z_shares_all, loss_dict_avg


    def train_cf_prop(self, server_logits_bank=None, base_alpha=0.5, T=3.0):
        """
        [修改版] 分类器训练
        1. 已移除动态权重，固定使用 base_alpha
        """
        self.freeze(self.ae_model)
        self.cf_model.train()

        loss_dict_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
            total_correct, total_samples = 0, 0
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
                    z_share, z_spec, out = self.ae_model.encode(x_m, m)
                    z_latent = torch.cat([z_share, z_spec], dim=-1)
                    latents.append(z_latent)

                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()

                logits = self.cf_model(latent_cat)

                # === 1. CE Loss ===
                ce_loss = self.cls_loss_fn(logits, y_batch)

                # === 2. 蒸馏部分 ===
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
                        # KL Loss
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
    
    def train_cf_prop_dyn(self, server_logits_bank=None, base_alpha=0.5, T=3.0):
        """
        [优化版] 分类器训练
        1. 动态蒸馏权重 (Entropy-based Dynamic Alpha)
        """
        self.freeze(self.ae_model)
        self.cf_model.train()

        loss_dict_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
            total_correct, total_samples = 0, 0
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
                    z_share, z_spec, out = self.ae_model.encode(x_m, m)
                    z_latent = torch.cat([z_share, z_spec], dim=-1)
                    latents.append(z_latent)

                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()

                logits = self.cf_model(latent_cat)

                # === 1. CE Loss ===
                ce_loss = self.cls_loss_fn(logits, y_batch)

                # === 2. 动态蒸馏 ===
                kd_loss = 0.0
                alpha = base_alpha

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

                        # --- 计算动态权重 (Dynamic Alpha) ---
                        with torch.no_grad():
                            student_probs = F.softmax(logits, dim=1)
                            entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1)
                            max_entropy = np.log(logits.size(1))
                            norm_entropy = entropy / max_entropy
                            dynamic_alpha = torch.clamp(norm_entropy, 0.1, 0.9).mean().item()

                        alpha = dynamic_alpha

                        # KL Loss
                        p_s = F.log_softmax(logits / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)
                        kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
                    else:
                        alpha = 0.0

                # === 总损失 ===
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

    def test(self, eval_win=EVAL_WIN):  # 确保覆盖 UserBase 的 test

        self.ae_model.eval()
        self.cf_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 确保测试集按序列切分 (假设 self.test_data 已加载)
        # 如果 self.test_data 已经是切分好的 tensor 字典，直接用
        # 这里沿用 make_seq_batch2 的逻辑，或者直接遍历

        # 假设 test_data 是 raw dict: {'acc': np.array, 'y': np.array...}
        # 我们需要滑窗测试
        y_test = self.test_data["y"]
        seq_len = len(y_test)

        with torch.no_grad():
            for start in range(0, seq_len - eval_win, eval_win):
                end = start + eval_win

                # 构造 Batch
                batch_x = {}
                for m in self.modalities:
                    # [1, Win, D] (模拟 Batch=1)
                    d = self.test_data[m][start:end]
                    batch_x[m] = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(self.device)

                y_batch = torch.tensor(y_test[start:end], dtype=torch.long).to(self.device)
                # 如果标签是按时间步的，展平；如果是 seq 级的，取一个
                # 假设是时间步级
                y_batch = y_batch.view(-1)

                # --- Forward ---
                latents = []
                for m in self.modalities:
                    # === 关键修复点：接收 3 个返回值 ===
                    z_share, z_spec, _ = self.ae_model.encode(batch_x[m], m)

                    # 拼接 Share 和 Spec 用于分类 (与训练时保持一致)
                    z_cat = torch.cat([z_share, z_spec], dim=-1)
                    latents.append(z_cat)

                # 拼接多模态特征
                if len(latents) > 0:
                    latent_cat = torch.cat(latents, dim=-1)  # [1, Win, D_total]

                    # 展平送入分类器 [Win, D_total]
                    # 或者是 [1*Win, D_total]
                    logits = self.cf_model(latent_cat)

                    # 调整 logits 维度以匹配 y_batch
                    if logits.dim() == 3:
                        logits = logits.view(-1, logits.shape[-1])

                    # 计算 Loss
                    loss = self.cls_loss_fn(logits, y_batch)
                    total_loss += loss.item() * y_batch.size(0)

                    # 计算 Acc
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return avg_acc, avg_loss
