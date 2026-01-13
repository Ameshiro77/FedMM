import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *
from FLAlgorithms.trainmodel.losses import *
from FLAlgorithms.config import *

class UserNoDis(User):
    def __init__(self, client_id, train_data, test_data, public_data, ae_model, cf_model,
                 modalities, batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1, args=None):
        super().__init__(client_id, train_data, test_data, public_data, ae_model, cf_model,
                         modalities, batch_size, learning_rate,
                         beta, lamda, local_epochs, label_ratio)

        self.client_align_weight = args.client_align_weight
        self.client_reg_weight = args.client_reg_weight
        self.client_logits_weight = args.client_logits_weight

    def upload_features(self):
        """上传特征，必须压扁为 2D"""
        self.ae_model.eval()
        z_dict = {m: None for m in self.modalities}
        modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
        
        for m in self.modalities:
            x_m = torch.from_numpy(modalities_seq[m]).to(self.device)
            # SplitAE.encode 返回 (mid, final)
            # final 是 [B, T, D]
            _, z_seq = self.ae_model.encode(x_m, m)
            
            # 【关键】压缩时间维度，变为 [B, D]，避免服务端 MMD 报错
            z_dict[m] = z_seq.mean(dim=1).detach().cpu() 
        return z_dict

    def train_ae_nodis(self, global_anchors, pre_w):
        self.ae_model.train()
        self.cf_model.eval()
        loss_epoch = {'rec_loss': [], 'align_loss': [], 'reg_loss': []}

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
                rec_loss, align_loss, reg_loss = 0.0, 0.0, 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    
                    x_recon = self.ae_model(x_m, m)
                    
                    # 获取特征用于对齐
                    _, z_seq = self.ae_model.encode(x_m, m)
                    z_pooled = z_seq.mean(dim=1) # [B, D]

                    rec_loss += self.rec_loss_fn(x_recon, x_m)

                    if global_anchors is not None and isinstance(global_anchors, dict):
                        if m in global_anchors:
                            z_anchor = global_anchors[m].to(self.device)
                            # Cosine Sim between [B, D] and [D]
                            cos_sim = F.cosine_similarity(z_pooled, z_anchor, dim=-1)
                            align_loss += (1 - cos_sim.mean())

                if pre_w is not None:
                    for p_new, p_old in zip(self.ae_model.parameters(), pre_w):
                        reg_loss += torch.sum((p_new - p_old.to(self.device))**2)

                loss = rec_loss + \
                       self.client_align_weight * align_loss + \
                       self.client_reg_weight * reg_loss
                
                loss.backward()
                self.optimizer_ae.step()

                loss_epoch['rec_loss'].append(rec_loss.item())
                loss_epoch['align_loss'].append(align_loss.item())
                loss_epoch['reg_loss'].append(reg_loss.item())

        return {k: np.mean(v) for k, v in loss_epoch.items()}

    def train_cf_nodis(self, server_logits_bank, base_alpha=0.5, T=3.0):
        self.freeze(self.ae_model)
        self.cf_model.train()
        loss_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
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
                    # out: [Batch, Time, Dim]
                    _, out = self.ae_model.encode(x_m, m) 
                    latents.append(out)
                
                # [Batch, Time, Total_Dim]
                latent_cat = torch.cat(latents, dim=-1) 
                
                # 【关键修正】展平时间步：[Batch * Time, Total_Dim]
                latent_flat = latent_cat.view(-1, latent_cat.shape[-1])

                # 标签也展平：[Batch * Time]
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()
                
                # Logits: [Batch * Time, Classes]
                logits = self.cf_model(latent_flat)
                
                ce_loss = self.cls_loss_fn(logits, y_batch)

                kd_loss = 0.0
                alpha = 0.0
                if server_logits_bank is not None:
                    # 只有当 teacher_logits 也是 [B*T, C] 或者能广播时才能算
                    # 通常 logits bank 存的是 label -> logits (Avg)。
                    # 我们对每个样本查表
                    teacher_logits = []
                    y_np = y_batch.detach().cpu().numpy()
                    has_teacher = True
                    
                    # 这里的 y_np 已经是展平的了，所以是对每个时间步查 Teacher
                    for y in y_np:
                        if y not in server_logits_bank:
                            has_teacher = False
                            break
                        teacher_logits.append(server_logits_bank[y].unsqueeze(0))
                    
                    if has_teacher:
                        teacher_logits = torch.cat(teacher_logits, dim=0).to(self.device)
                        alpha = base_alpha
                        p_s = F.log_softmax(logits / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)
                        kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
                
                loss = (1 - alpha) * ce_loss + alpha * kd_loss
                loss.backward()
                self.optimizer_cf.step()
                
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == y_batch).sum().item()
                loss_epoch['cf_loss'].append(loss.item())
                loss_epoch['cf_acc'].append(correct / y_batch.size(0))
        
        self.unfreeze(self.ae_model)
        return {k: np.mean(v) for k, v in loss_epoch.items()}

    def train_cf_nodis_dyn(self, server_logits_bank, base_alpha=0.5, T=3.0):
        self.freeze(self.ae_model)
        self.cf_model.train()
        loss_epoch = {'cf_loss': [], 'cf_acc': []}

        for ep in range(self.local_epochs):
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
                    _, out = self.ae_model.encode(x_m, m) 
                    latents.append(out)
                
                latent_cat = torch.cat(latents, dim=-1) 
                # 【修正】展平 [B*T, D]
                latent_flat = latent_cat.view(-1, latent_cat.shape[-1])
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()
                
                logits = self.cf_model(latent_flat)
                ce_loss = self.cls_loss_fn(logits, y_batch)

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
                        
                        # Entropy logic
                        with torch.no_grad():
                            student_probs = F.softmax(logits, dim=1)
                            entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1)
                            max_entropy = np.log(logits.size(1))
                            norm_entropy = entropy / max_entropy
                            dynamic_alpha = torch.clamp(norm_entropy, 0.1, 0.9).mean().item()
                        
                        alpha = dynamic_alpha
                        p_s = F.log_softmax(logits / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)
                        kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
                
                loss = (1 - alpha) * ce_loss + alpha * kd_loss
                loss.backward()
                self.optimizer_cf.step()
                
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == y_batch).sum().item()
                loss_epoch['cf_loss'].append(loss.item())
                loss_epoch['cf_acc'].append(correct / y_batch.size(0))
        
        self.unfreeze(self.ae_model)
        return {k: np.mean(v) for k, v in loss_epoch.items()}

    def test(self, eval_win=100):
        self.ae_model.eval()
        self.cf_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        y_test = self.test_data["y"]
        seq_len = len(y_test)

        with torch.no_grad():
            for start in range(0, seq_len - eval_win, eval_win):
                end = start + eval_win
                
                batch_x = {}
                for m in self.modalities:
                    d = self.test_data[m][start:end]
                    batch_x[m] = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(self.device)

                # y_batch: [Win] -> 展平后大小为 eval_win
                y_batch = torch.tensor(y_test[start:end], dtype=torch.long).to(self.device)
                y_batch = y_batch.view(-1)

                latents = []
                for m in self.modalities:
                    # encode 返回 (out, out1)。out 是 [1, Win, D]
                    out, _ = self.ae_model.encode(batch_x[m], m)
                    latents.append(out)
                
                if len(latents) > 0:
                    latent_cat = torch.cat(latents, dim=-1) # [1, Win, Total_D]
                    
                    # 【修正】展平 [Win, Total_D]
                    latent_flat = latent_cat.view(-1, latent_cat.shape[-1])
                    
                    logits = self.cf_model(latent_flat)
                    
                    loss = self.cls_loss_fn(logits, y_batch)
                    total_loss += loss.item() * y_batch.size(0)

                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return avg_acc, avg_loss
    def test(self, eval_win=100):
        self.ae_model.eval()
        self.cf_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        y_test = self.test_data["y"]
        seq_len = len(y_test)

        with torch.no_grad():
            for start in range(0, seq_len - eval_win, eval_win):
                end = start + eval_win
                
                batch_x = {}
                for m in self.modalities:
                    d = self.test_data[m][start:end]
                    batch_x[m] = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(self.device)

                y_batch = torch.tensor(y_test[start:end], dtype=torch.long).to(self.device)
                y_batch = y_batch.view(-1)

                latents = []
                for m in self.modalities:
                    _, z_seq = self.ae_model.encode(batch_x[m], m)
                    latents.append(z_seq.squeeze(0)) # [Win, Dim]
                
                if len(latents) > 0:
                    latent_cat = torch.cat(latents, dim=-1) # [Win, Total_D]
                    logits = self.cf_model(latent_cat)
                    
                    loss = self.cls_loss_fn(logits, y_batch)
                    total_loss += loss.item() * y_batch.size(0)

                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return avg_acc, avg_loss