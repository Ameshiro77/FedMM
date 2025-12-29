from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.userfedab import UserNoDis # 确保这里引用的 UserNoDis 是正确的
# from FLAlgorithms.users.userbase import User
from utils.model_utils import *
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.users.userprop import * # 依然需要 UserProp 里的一些工具或配置
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
        self.users = []
        
        # === Server Side Parameters (Aligning with FedProp) ===
        self.server_dist_weight = args.server_dist_weight
        self.server_align_weight = args.server_align_weight

        # 1. 投影头 (用于语义对齐)
        self.proj_heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(rep_size, rep_size), nn.ReLU(), nn.Linear(rep_size, rep_size))
            for m in self.modalities_server
        }).to(self.device)
        
        # 2. 注意力权重 (如果你使用了注意力融合)
        self.attn_w = nn.Parameter(torch.randn(rep_size, 1).to(self.device) * 0.1).to(self.device)
        nn.init.xavier_uniform_(self.attn_w)
        
        # 3. 模态权重 (用于加权融合，如果使用)
        self.modality_weight = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in self.modalities_server
        })

        # 4. 服务端分类器
        self.cf_model_server = MLP(rep_size, n_classes).to(self.device)

        # 5. Loss History
        self.loss_history = {
            'rec_loss': [],
            'orth_loss': [], # UserNoDis 可能没有这个，但为了格式统一保留
            'align_loss': [],
            'reg_loss': []
        }
        self.server_loss_history = {
            'distill_loss': [],
            'align_loss': [],
            'cls_loss': [],
            'total_loss': [],
            'proto_loss': []
        }
        
        # === User Initialization ===
        for i in range(self.total_users):
            client_modals = modalities[i]
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
                num_modals = len(client_modals)
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}
                num_modals = 1
                
            input_size = list(input_sizes_client.values())[0]

            # === 关键点 1: 使用 SplitLSTMAutoEncoder (标准 AE，无解耦) ===
            # 注意：SplitLSTMAutoEncoder 通常是指分模态编码，但不分 share/private
            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size 
            )
            
            # === 关键点 2: 分类器输入维度 ===
            # 对于 NoDis，通常是将所有模态特征拼接放入分类器
            # 维度 = rep_size * 模态数量
            client_cf = MLP(rep_size * num_modals, n_classes)
            
            if pfl:
                user = UserNoDis(
                    i, self.clients_train_data_list[i], self.clients_test_data_list[i], self.public_data, 
                    client_ae, client_cf, client_modals, batch_size, learning_rate,
                    beta, lamda, local_epochs, label_ratio, args=args
                )
            else:
                user = UserNoDis(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data, 
                    client_ae, client_cf, client_modals, batch_size, learning_rate,
                    beta, lamda, local_epochs, label_ratio, args=args
                )
            self.users.append(user)
    
    def train_server(self, z_shares_all_list, prototypes_weights, glob_iter):
        """
        服务端蒸馏 + 对齐 + 分类联合训练
        逻辑与 FedProp 保持一致，但处理的是未解耦的 z
        """
        self.cf_model_server.train()
        self.ae_model_server.train()

        # =========================================================================================
        # === Step 1. 聚合同模态客户端特征 ===
        z_shares_all = {}
        for client_dict in z_shares_all_list:
            for mod, z_share in client_dict.items():
                z_shares_all.setdefault(mod, []).append(z_share)

        # 可视化 (可选)
        if glob_iter % 10 == 0:
            pass 

        # === Step 2. 使用服务端数据进行训练 ===
        modalities_seq, labels = make_seq_batch2(self.server_train_data, self.batch_size)
        seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]
        z_fuse_all = []
        
        for epoch in range(1):
            idx_end = 0
            while idx_end < seq_len_batch:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len_batch)

                self.optimizer_ae.zero_grad()
                self.optimizer_cf.zero_grad()

                z_m_all = {}
                dist_loss, align_loss, cls_loss = 0.0, 0.0, 0.0

                # === Step 3. 计算每个模态的server特征 & 蒸馏 ===
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z_m = self.ae_model_server.encode(x_m, m)      # [B, T, D]
                    z_m_all[m] = z_m
                    z_m_pooled = z_m.mean(dim=1)                      # [B, D]

                    # ====== 准备客户端特征 bank ======
                    # 这里的 z_shares_all[m] 来自 UserNoDis，是完整的特征 z
                    z_s = F.normalize(z_m_pooled, dim=-1)             # [B, D]
                    z_c = F.normalize(torch.cat(z_shares_all[m], dim=0).to(self.device), dim=-1)  # [N_total, D]

                    # ====== 计算 MMD Loss (与 FedProp 一致) ======
                    mmd_loss_val = compute_mmd(z_s, z_c)
                    dist_loss += mmd_loss_val

                # === Step 4. 模态间语义对齐 (与 FedProp 一致) ===
                z_proj_dict = {m: self.proj_heads[m](z_m_all[m]) for m in z_m_all.keys()}
                align_loss = contrastive_modality_align(z_proj_dict)

                # === Step 5. 分类训练 ===
                # 使用 FusionNet 融合特征 (确保 Server 基类有 fusionNet 或在 init 定义)
                z_fuse = self.fusionNet(z_m_all)  # B,W,D
                z_fuse_all.append(z_fuse)
                
                y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device).flatten().long()
                logits = self.cf_model_server(z_fuse)
                
                cls_loss = self.cls_loss_fn(logits, y_true)

                # === Logits Bank 更新 ===
                for cls_id in y_true.unique():
                    cls_mask = (y_true == cls_id)
                    if cls_mask.sum() > 0:
                        logits_cls = logits[cls_mask].mean(dim=0)
                        if not hasattr(self, "logits_bank"):
                            self.logits_bank = {}
                        if cls_id.item() not in self.logits_bank:
                            self.logits_bank[cls_id.item()] = logits_cls.clone().detach()
                        else:
                            self.logits_bank[cls_id.item()] = (
                                0.5 * self.logits_bank[cls_id.item()] + 0.5 * logits_cls.clone().detach()
                            )

                # === Step 6. 总损失 ===
                total_loss = (
                    self.server_dist_weight * dist_loss +
                    self.server_align_weight * align_loss +
                    1.0 * cls_loss
                )

                total_loss.backward()
                self.optimizer_ae.step()
                self.optimizer_cf.step()

            print(f"[Server Train] Distill={dist_loss:.4f} Align={align_loss:.4f} Cls={cls_loss:.4f}")

        self.server_loss_history['distill_loss'].append(dist_loss.item())
        self.server_loss_history['cls_loss'].append(cls_loss.item())
        self.server_loss_history['total_loss'].append(total_loss.item())

        # 返回融合后的特征均值，用于可能的客户端对齐 (UserNoDis 中是否使用取决于具体实现)
        # 以及 logits_bank 用于 KD
        return torch.cat(z_fuse_all, dim=1).mean(dim=1).detach(), None, self.logits_bank
    
    def train(self):
        """
        完整训练循环 (修复版)
        """
        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- Global Round [{glob_iter+1}/{self.num_glob_iters}] (NoDis Ablation) ---")

            # Step 1: 客户端上传特征
            self.selected_users = self.select_users(glob_iter, self.num_users)
            z_shares_all = [user.upload_shares() for user in self.selected_users]
            
            # Step 2: 服务器训练
            z_global, global_prototypes, server_logits = self.train_server(z_shares_all, None, glob_iter)
            
            # Step 3: 客户端 AE 训练
            epoch_losses_clients = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}
            
            for user in self.selected_users:
                # === 【修复点】获取更新前的权重 pre_w ===
                # 这用于计算正则化损失 (防止模型漂移过远)
                pre_w = [p.clone().detach().cpu() for p in user.ae_model.parameters()]
                
                # === 【修复点】将 pre_w 传给 train_ae_prop ===
                loss_dict = user.train_ae_prop(z_global, global_prototypes, pre_w) 
                
                print(f"[Client {user.client_id}] AE => Recon={loss_dict.get('rec_loss', 0):.4f}")

                # 记录损失
                if not hasattr(user, 'loss_history'):
                    user.loss_history = {k: [] for k in epoch_losses_clients.keys()}
                
                for k in epoch_losses_clients.keys():
                    if k in loss_dict:
                        user.loss_history[k].append(loss_dict[k])
                        epoch_losses_clients[k].append(loss_dict[k])

            # Step 4: 客户端 CF 训练
            cf_losses_clients = {'cf_loss': [], 'cf_acc': []}
            for user in self.selected_users:
                # 使用 KD 训练分类器
                if self.args.client_logits_weight > 0.001:
                    cf_loss_dict = user.train_cf_prop(server_logits)
                else:
                    cf_loss_dict = user.train_cf_prop(None)
                
                print(f"[Client {user.client_id}] CF => Loss={cf_loss_dict['cf_loss']:.4f}, Acc={cf_loss_dict['cf_acc']:.4f}")

                # 记录损失
                if not hasattr(user, 'cf_loss_history'):
                    user.cf_loss_history = {k: [] for k in cf_losses_clients.keys()}
                
                for k in cf_losses_clients.keys():
                    user.cf_loss_history[k].append(cf_loss_dict[k])
                    cf_losses_clients[k].append(cf_loss_dict[k])

            # Step 5: 记录全局平均损失
            for k in epoch_losses_clients.keys():
                if epoch_losses_clients[k]:
                    self.loss_history.setdefault(k, []).append(np.mean(epoch_losses_clients[k]))
            for k in cf_losses_clients.keys():
                if cf_losses_clients[k]:
                    self.loss_history.setdefault(k, []).append(np.mean(cf_losses_clients[k]))

            # Step 6: 服务端测试
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)

        # 训练结束，测试客户端并保存
        if self.args.ablation:
            client_accs, avg_client_acc, avg_modality_acc = self.test_clients()
            return self.rs_glob_acc, avg_client_acc, avg_modality_acc
        else:
            self.save_results()
            self.save_results()

    def test_server(self):
        """服务器端测试集评估 (完全复用 FedProp 代码)"""
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
                    _, hidden_seq = self.ae_model_server.encode(batch_x[m], m)  
                    z_m_all[m] = hidden_seq    

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

        print(f"Server Test - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, avg_acc, f1