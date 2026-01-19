from FLAlgorithms.users.userbase import User
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.users.userfedab import *
import json
import codecs
from FLAlgorithms.trainmodel.ae_model import *
from FLAlgorithms.trainmodel.losses import *
from torch import nn, optim
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn.functional as F
import os

class FedNoDis(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False, args=None):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        self.total_users = len(modalities)
        self.pfl = pfl
        self.args = args
        self.server_dist_weight = args.server_dist_weight
        self.server_align_weight = args.server_align_weight
        self.rep_size = rep_size

        # === 1. 模态权重 ===
        self.modality_weight = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in self.modalities_server
        })

        # === 2. 投影头 ===
        self.proj_heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(rep_size, rep_size), nn.ReLU(), nn.Linear(rep_size, rep_size))
            for m in self.modalities_server
        }).to(self.device)

        # === 3. 服务端模型 (SplitLSTMAutoEncoder) ===
        self.ae_model_server = SplitLSTMAutoEncoder(
            input_sizes=input_sizes, 
            representation_size=rep_size,
            num_layers=1,
            batch_first=True
        ).to(self.device)

        self.cf_model_server = MLP(rep_size, n_classes).to(self.device)

        # === 4. 初始化 UserNoDis ===
        for i in range(self.total_users):
            client_modals = modalities[i]
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size,
                num_layers=1,
                batch_first=True
            )
            
            # 消融实验：拼接所有模态特征送入分类器
            num_client_modals = len(client_modals) if isinstance(client_modals, list) else 1
            client_cf = MLP(rep_size * num_client_modals, n_classes)

            if pfl:
                user = UserNoDis(
                    i, self.clients_train_data_list[i],
                    self.clients_test_data_list[i],
                    self.public_data, client_ae, client_cf, client_modals, batch_size, learning_rate, beta, lamda,
                    local_epochs, label_ratio=label_ratio, args=args)
            else:
                user = UserNoDis(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data, client_ae, client_cf,
                    client_modals, batch_size, learning_rate, beta, lamda,
                    local_epochs, label_ratio=label_ratio, args=args)
            
            user.info()
            self.users.append(user)

        self.loss_history = {'rec_loss': [], 'align_loss': [], 'reg_loss': []}
        self.server_loss_history = {'distill_loss': [], 'align_loss': [], 'cls_loss': [], 'total_loss': []}

    def train_server(self, z_features_all_list, prototypes_weights, glob_iter):
        self.cf_model_server.train()
        self.ae_model_server.train()

        z_features_all = {}
        for client_dict in z_features_all_list:
            for mod, z_feat in client_dict.items():
                z_features_all.setdefault(mod, []).append(z_feat)

        modalities_seq, labels = make_seq_batch2(self.server_train_data, self.batch_size)
        seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]

        for epoch in range(1):
            idx_end = 0
            while idx_end < seq_len_batch:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len_batch)

                self.optimizer_ae.zero_grad()
                self.optimizer_cf.zero_grad()

                z_m_all = {} # 存储 [B, T, D] 用于对齐和Fusion
                dist_loss, align_loss, cls_loss = 0.0, 0.0, 0.0

                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    
                    # SplitAE.encode 返回 (out_mid, out_final)
                    # out_final 是 [B, T, D]
                    _, z_seq = self.ae_model_server.encode(x_m, m)
                    
                    z_m_all[m] = z_seq 
                    
                    z_m_pooled = z_seq.mean(dim=1) 

                    if m in z_features_all:
                        z_s = F.normalize(z_m_pooled, dim=-1)
                        z_c = F.normalize(torch.cat(z_features_all[m], dim=0).to(self.device), dim=-1)
                        dist_loss += compute_mmd(z_s, z_c)

                # === Modality Alignment (使用 Sequence 对齐 或 Pooled 对齐) ===
                z_proj_dict = {m: self.proj_heads[m](z_m_all[m]) for m in z_m_all.keys()}
                align_loss = contrastive_modality_align(z_proj_dict)

                # === Classification ===
                z_fuse = self.fusionNet(z_m_all) 
                
                y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device).flatten().long()
                logits = self.cf_model_server(z_fuse)
                
                cls_loss = self.cls_loss_fn(logits, y_true)

                # Update Logits Bank
                for cls_id in y_true.unique():
                    cls_mask = (y_true == cls_id)
                    if cls_mask.sum() > 0:
                        logits_cls = logits[cls_mask].mean(dim=0)
                        if not hasattr(self, "logits_bank"): self.logits_bank = {}
                        if cls_id.item() not in self.logits_bank:
                            self.logits_bank[cls_id.item()] = logits_cls.clone().detach()
                        else:
                            self.logits_bank[cls_id.item()] = 0.5 * self.logits_bank[cls_id.item()] + 0.5 * logits_cls.clone().detach()

                total_loss = (
                    self.server_dist_weight * dist_loss +
                    self.server_align_weight * align_loss +
                    1.0 * cls_loss
                )

                total_loss.backward()
                self.optimizer_ae.step()
                self.optimizer_cf.step()

            print(f"[Server NoDis] Distill={dist_loss:.4f} Align={align_loss:.4f} Cls={cls_loss:.4f}")
            
            self.server_loss_history['distill_loss'].append(dist_loss.item())
            self.server_loss_history['align_loss'].append(align_loss.item())
            self.server_loss_history['cls_loss'].append(cls_loss.item())
            self.server_loss_history['total_loss'].append(total_loss.item())

        # Re-extract Global Anchors
        self.ae_model_server.eval()
        h_accumulators = {m: [] for m in self.modalities_server}
        with torch.no_grad():
            idx_start = 0
            win_len_extract = 32
            seq_len_total = modalities_seq[self.modalities_server[0]].shape[1]
            while idx_start < seq_len_total:
                idx_end = min(idx_start + win_len_extract, seq_len_total)
                if idx_end - idx_start < 8: break
                
                for m in self.modalities_server:
                    x_chunk = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z_seq = self.ae_model_server.encode(x_chunk, m)
                    # 同样做 mean pooling，得到 [B, D]
                    h_accumulators[m].append(z_seq.mean(dim=1))
                
                idx_start += win_len_extract

        global_anchors = {}
        for m in self.modalities_server:
            if len(h_accumulators[m]) > 0:
                global_anchors[m] = torch.cat(h_accumulators[m], dim=0).mean(dim=0).detach().cpu()
            else:
                global_anchors[m] = torch.zeros(self.rep_size).cpu()

        return global_anchors, None, self.logits_bank

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- [NoDis] Global Round [{glob_iter+1}/{self.num_glob_iters}] ---")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            
            z_features_all = [user.upload_features() for user in self.selected_users]
            # prototypes_weights = [(user.upload_prototype(), user.get_prototype_weight()) 
            #                       for user in self.selected_users]

            z_global, _, server_logits = self.train_server(z_features_all, None, glob_iter)

            epoch_losses = {'rec_loss': [], 'align_loss': [], 'reg_loss': []}
            
            for user in self.selected_users:
                pre_w = [p.clone().detach().cpu() for p in user.ae_model.parameters()]
                loss_dict = user.train_ae_nodis(z_global, pre_w) 
                
                if not hasattr(user, 'loss_history'):
                    user.loss_history = {k: [] for k in ['rec_loss', 'align_loss', 'reg_loss']}
                
                for k, v in loss_dict.items():
                    user.loss_history[k].append(v)
                    epoch_losses[k].append(v)

            cf_losses = {'cf_loss': [], 'cf_acc': []}
            for user in self.selected_users:
                if self.args.client_logits_weight > 0.001:
                    cf_loss_dict = user.train_cf_nodis_dyn(server_logits)
                else:
                    cf_loss_dict = user.train_cf_nodis(server_logits)
                
                if not hasattr(user, 'cf_loss_history'):
                    user.cf_loss_history = {k: [] for k in ['cf_loss', 'cf_acc']}

                for k, v in cf_loss_dict.items():
                    user.cf_loss_history[k].append(v)
                    cf_losses[k].append(v)

            for k in epoch_losses:
                self.loss_history[k].append(np.mean(epoch_losses[k]))
            
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)
            
        if self.args.ablation:
            client_accs, avg_client_acc, avg_modality_acc = self.test_clients(save=False)
            print("Test accuracy: ", avg_client_acc)
            return self.rs_glob_acc, avg_client_acc, avg_modality_acc


        else:
            if self.pfl:
                save_dir = f"./results/pfl/{self.dataset}/"
            else:
                save_dir = f"./results/{self.dataset}/"
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, "loss_server.json")

            # 收集所有客户端的 Loss (遍历 self.users 而不是 self.selected_users 以获取完整历史)
            client_loss_all = {}
            for user in self.users:
                if hasattr(user, 'loss_history'):
                    client_loss_all[user.client_id] = {
                        "ae_loss": user.loss_history,
                        "cf_loss": getattr(user, 'cf_loss_history', {})
                    }
            
            all_losses = {
                "global_losses": self.loss_history,       # AE 与 CF 全局平均损失
                "global_train_loss": self.rs_train_loss,  # server test loss
                "global_acc": self.rs_glob_acc,
                "global_f1": self.rs_glob_f1,
                "client_losses": client_loss_all          # 每个客户端
            }
            
            with open(save_path, "w") as f:
                json.dump(all_losses, f, indent=4)
            print(f"[Saved] Loss JSON => {save_path}")

            client_accs, avg_client_acc, avg_modality_acc = self.test_clients()
            self.save_results()
            return None
        # accs = self.test_clients()
        # print("Test accuracy: ", accs)
        # self.save_results()

    def test_server(self):
        self.ae_model_server.eval()
        self.cf_model_server.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        eval_win = EVAL_WIN
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for start in range(0, len(self.test_data["y"]) - eval_win + 1, eval_win):
                batch_x = {
                    m: torch.tensor(
                        self.test_data[m][start:start + eval_win],
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    for m in self.modalities_server
                }
                batch_y = torch.tensor(
                    self.test_data["y"][start:start + eval_win],
                    dtype=torch.long, device=self.device
                )

                z_m_all = {}
                for m in self.modalities_server:
                    _, z_seq = self.ae_model_server.encode(batch_x[m], m)
                    z_m_all[m] = z_seq

                z_fuse = self.fusionNet(z_m_all)
                outputs = self.cf_model_server(z_fuse)

                loss = self.cls_loss_fn(outputs, batch_y)
                preds = torch.argmax(outputs, dim=1)

                total_loss += loss.item() * batch_y.size(0)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        print(f"Server NoDis Test - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, F1: {f1:.4f}")
        return avg_loss, avg_acc, f1