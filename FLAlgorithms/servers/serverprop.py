from FLAlgorithms.users.userbase import User
# from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.users.userprop import *
import json
import codecs
from FLAlgorithms.trainmodel.ae_model import *
from torch import nn, optim
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class FedProp(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times)

        self.total_users = len(modalities)

        self.attn_temperature = 1.0  # 可调
        self.attn_w = nn.Parameter(torch.randn(rep_size, 1).to(self.device) * 0.1).to(self.device)
        self.proj_heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(rep_size, rep_size), nn.ReLU(), nn.Linear(rep_size, rep_size))
            for m in self.modalities_server
        }).to(self.device)
        nn.init.xavier_uniform_(self.attn_w)

        for i in range(self.total_users):

            client_modals = modalities[i]
            if isinstance(client_modals, list):  # 支持多模态客户端
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:  # 单模态
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            input_size = list(input_sizes_client.values())[0]
            # 实例化该客户端的 AE
            # client_ae = SplitLSTMAutoEncoder(
            #     input_sizes=input_sizes_client,
            #     representation_size=rep_size
            # )
            client_ae = DisentangledLSTMAutoEncoder(
                input_size=input_size,
                representation_size=rep_size,
                shared_size=rep_size,
                specific_size=rep_size,
            )

            client_cf = MLP(rep_size*2, n_classes)
            user = UserProp(
                i, self.clients_train_data_list[i], self.test_data, self.public_data, client_ae, client_cf,
                client_modals, batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio=label_ratio)

            user.info()

            print("client", i, "modals:", client_modals)
            self.users.append(user)

        # exit()
        # 用于记录每轮各客户端损失
        self.loss_history = {
            'rec_loss': [],
            'orth_loss': [],
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

    def _visualize_modal_distributions(self, z_shares_all, glob_iter):
        all_feats, all_labels = [], []
        for m, clients_list in z_shares_all.items():  # 遍历模态
            for c_idx, z in enumerate(clients_list):  # 遍历客户端
                z_np = z.detach().cpu().numpy()  # [D]
                all_feats.append(z_np)
                all_labels.append(f"{m}_client{c_idx}")

        all_feats = np.stack(all_feats)  # [num_modalities * num_clients, D]

        # TSNE降维可视化
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        z_emb = tsne.fit_transform(all_feats)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=z_emb[:, 0], y=z_emb[:, 1], hue=all_labels, palette="tab10", s=80)
        plt.title("Modal-Client Feature Distribution", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"results/dist/modal_{self.dataset}_distributions_{glob_iter}.png")
        plt.close()

    def _visualize_local_prototypes_3d(self, prototypes_weights, glob_iter):
        """
        3D 可视化每个客户端的 prototype 分布。
        每个点 = 某客户端的某一类原型
        颜色 = 类别
        标记形状 = 客户端
        """
        all_feats, all_labels, all_clients = [], [], []

        for client_idx, (proto_dict, _) in enumerate(prototypes_weights):
            for cls, proto in proto_dict.items():
                z_np = proto.detach().cpu().numpy().reshape(-1)
                all_feats.append(z_np)
                all_labels.append(int(cls))
                all_clients.append(client_idx)

        all_feats = np.stack(all_feats)

        # === t-SNE 降维到3D ===
        tsne = TSNE(n_components=3, perplexity=5, random_state=42)
        z_emb = tsne.fit_transform(all_feats)

        # === 绘制3D图 ===
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        num_classes = len(np.unique(all_labels))
        palette = sns.color_palette("tab10", num_classes)
        markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

        for i in range(len(z_emb)):
            c = all_labels[i]
            client = all_clients[i]
            ax.scatter(
                z_emb[i, 0], z_emb[i, 1], z_emb[i, 2],
                color=palette[c % num_classes],
                marker=markers[client % len(markers)],
                s=80, edgecolor="k", linewidth=0.3,
            )

        ax.set_title(f"3D Prototype Distribution (Iter {glob_iter})", fontsize=13)
        ax.set_xlabel("TSNE-1")
        ax.set_ylabel("TSNE-2")
        ax.set_zlabel("TSNE-3")
        plt.tight_layout()
        plt.savefig(f"results/dist/prototypes3D_{self.dataset}_{glob_iter}.png", dpi=300)
        plt.close()
    
    def train_server(self, z_shares_all_list, prototypes_weights, glob_iter):
        """
        服务端蒸馏 + 对齐 + 分类联合训练
        """
        self.cf_model_server.train()
        self.ae_model_server.train()

        # =========================================================================================
        # === Step 1. 聚合同模态客户端特征 ===
        
        # z_shares
        z_shares_all = {}
        for client_dict in z_shares_all_list:
            for mod, z_share in client_dict.items():
                z_shares_all.setdefault(mod, []).append(z_share)

        if glob_iter % 10 == 0:
            pass
            self._visualize_local_prototypes_3d(prototypes_weights, glob_iter)
            # self._visualize_modal_distributions(z_shares_all, glob_iter)

        z_modal_clients = {}
        for m, zs in z_shares_all.items():
            z_means = torch.stack(zs, dim=0)  # [num_clients_m, B, D_m]
            z_modal_clients[m] = z_means.mean(dim=0).to(self.device)  # FedAvg
            
        # prototypes
        total_counts = {}
        for _, weight_dict in prototypes_weights:
            for cls, count in weight_dict.items():
                total_counts[cls] = total_counts.get(cls, 0) + count

        # 初始化全局原型容器
        global_prototypes = {}

        for proto_dict, weight_dict in prototypes_weights:
            for cls, proto in proto_dict.items():
                if cls not in total_counts or total_counts[cls] == 0:
                    continue
                weight = weight_dict[cls] / total_counts[cls]
                if cls not in global_prototypes:
                    global_prototypes[cls] = proto * weight
                else:
                    global_prototypes[cls] += proto * weight
                    
    
        # =========================================================================================

        # === Step 2. 使用服务端数据进行训练 ===
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

                z_m_all = {}
                dist_loss, align_loss, cls_loss, proto_loss = 0.0, 0.0, 0.0, 0.0

                # === Step 3. 计算每个模态的server特征 & 蒸馏 ===
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z_m = self.ae_model_server.encode(x_m, m)
                    z_m_all[m] = z_m
                    z_m_pooled = z_m.mean(dim=1)  # [batch_size, feature_dim] - 平均池化

                    # 2. COSINE
                    print(z_m_pooled.shape, z_modal_clients[m].shape)
                    z_m_pooled_norm = F.normalize(z_m_pooled, dim=-1)
                    z_client_norm = F.normalize(z_modal_clients[m], dim=-1)

                    dist_loss += 1 - torch.mean(torch.sum(z_m_pooled_norm * z_client_norm, dim=-1))

                    # 3. mmd
                    # def mmd_loss(x, y):
                    #     xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
                    #     rx = xx.diag().unsqueeze(0).expand_as(xx)
                    #     ry = yy.diag().unsqueeze(0).expand_as(yy)
                    #     Kxx = torch.exp(-0.5 * (rx.t() + rx - 2*xx))
                    #     Kyy = torch.exp(-0.5 * (ry.t() + ry - 2*yy))
                    #     Kxy = torch.exp(-0.5 * (rx.t() + ry - 2*xy))
                    #     return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

                    # dist_loss += mmd_loss(z_m_pooled, z_modal_clients[m])

                    # 4.constra
                    # Normalize
                    # z_s = F.normalize(z_m_pooled, dim=-1)
                    # z_c = F.normalize(z_modal_clients[m], dim=-1)

                    # # 相似度矩阵
                    # sim = torch.matmul(z_s, z_c.T) / tau   # [B, B]
                    # labels = torch.arange(B).to(sim.device)
                    # loss_contrast = F.cross_entropy(sim, labels)
                    # dist_loss += loss_contrast

                # === Step 4. 模态间语义对齐 ===
                # z_stack = torch.stack(list(z_m_all.values()), dim=0)  # [M, B,win, rep]\

                # z_center = z_stack.mean(dim=0)
                # for m in z_m_all.keys():
                #     align_loss += torch.nn.functional.mse_loss(z_m_all[m], z_center)

                # 对齐
                z_proj = [self.proj_heads[m](z_m_all[m]) for m in z_m_all.keys()]
                z_center = torch.stack(z_proj).mean(dim=0)
                align_loss = sum(F.mse_loss(z_proj[i], z_center) for i in range(len(z_proj)))

                # === Step 5. 分类训练 ===
                z_fuse = torch.cat([z for z in z_m_all.values()], dim=-1)
                y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device).flatten().long()

                logits = self.cf_model_server(z_fuse)
                # print(logits.shape,y_true.shape)
                cls_loss = self.cls_loss_fn(logits, y_true)

                    
                for cls_id in y_true.unique():
                    cls_mask = (y_true == cls_id)
                    if cls_mask.sum() > 0:
                        logits_cls = logits[cls_mask].mean(dim=0)
                        # 累加到全局平均表
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
                    0.1 * dist_loss +
                    0.1 * align_loss +
                    1.0 * cls_loss +
                    0.05 * proto_loss  #
                )
            
                total_loss.backward()
                self.optimizer_ae.step()
                self.optimizer_cf.step()

            print(f"[Server Train] Distill={dist_loss:.4f} Align={align_loss:.4f} Cls={cls_loss:.4f}")

            z_global_center = z_center.mean(dim=0).detach()  # 跨batch平均

        self.server_loss_history['distill_loss'].append(dist_loss.item())
        self.server_loss_history['align_loss'].append(align_loss.item())
        self.server_loss_history['cls_loss'].append(cls_loss.item())
        self.server_loss_history['total_loss'].append(total_loss.item())
        # self.server_loss_history['proto_loss'].append(proto_loss.item())

        print(z_global_center.shape, "111")
        return z_global_center.mean(dim=0), global_prototypes, self.logits_bank

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- Global Round [{glob_iter+1}/{self.num_glob_iters}] ---")

            # Step 1: 客户端提取 z_share
            self.selected_users = self.select_users(glob_iter, self.num_users)
            z_shares_all = [user.upload_shares() for user in self.selected_users]
            prototypes_weights = [ (user.upload_prototype(), user.get_prototype_weight()) for user in self.selected_users]

            # Step 2: 服务器训练
            z_global, global_prototypes, server_logits = self.train_server(z_shares_all, prototypes_weights, glob_iter)

            # Step 3: 客户端 AE 训练
            epoch_losses_clients = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}
            cf_losses_clients = {'cf_loss': [], 'cf_acc': []}

            for user in self.selected_users:
                pre_w = [p.clone().detach().cpu() for p in user.ae_model.parameters()]
                z_shares_new, loss_dict = user.train_ae_prop(z_global, global_prototypes, pre_w)
                print(f"[Client {user.client_id}] AE => Recon={loss_dict['rec_loss']:.4f}")

                # 初始化客户端损失字典（若不存在）
                if not hasattr(user, 'loss_history'):
                    user.loss_history = {k: [] for k in ['rec_loss', 'orth_loss', 'align_loss', 'reg_loss']}
                if not hasattr(user, 'cf_loss_history'):
                    user.cf_loss_history = {k: [] for k in ['cf_loss', 'cf_acc']}

                # 记录单客户端 AE 损失
                for k in epoch_losses_clients.keys():
                    user.loss_history[k].append(loss_dict[k])
                    epoch_losses_clients[k].append(loss_dict[k])  # 记录到全局

            # Step 4: 客户端 CF 训练
            for user in self.selected_users:
                cf_loss_dict = user.train_cf_prop(server_logits)
                print(f"[Client {user.client_id}] CF => Loss={cf_loss_dict['cf_loss']:.4f}, Acc={cf_loss_dict['cf_acc']:.4f}")

                # 记录单客户端 CF 损失
                for k in cf_losses_clients.keys():
                    user.cf_loss_history[k].append(cf_loss_dict[k])
                    cf_losses_clients[k].append(cf_loss_dict[k])  # 记录到全局

            # Step 5: 记录平均 AE/CF 损失
            for k in epoch_losses_clients.keys():
                self.loss_history.setdefault(k, []).append(np.mean(epoch_losses_clients[k]))
            for k in cf_losses_clients.keys():
                self.loss_history.setdefault(k, []).append(np.mean(cf_losses_clients[k]))

            # Step 6: 测试
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)

        # Step 7: 可视化 + 保存
        self.plot_losses(save_dir=f"./results/{self.dataset}/")
        self.plot_client_losses(save_dir=f"./results/{self.dataset}/")
        self.test_clients()
        self.save_results()

    def plot_losses(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # ---- 客户端平均损失 ----
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 调整为2x3布局
        axes = axes.flatten()

        loss_keys = ['rec_loss', 'orth_loss', 'align_loss', 'reg_loss', 'cf_loss', 'cf_acc']
        titles = ['Reconstruction Loss', 'Orthogonal Loss', 'Alignment Loss',
                  'Regularization Loss', 'Classification Loss', 'Classification Accuracy']

        for i, (k, title) in enumerate(zip(loss_keys, titles)):
            if k in self.loss_history and self.loss_history[k]:
                axes[i].plot(range(1, len(self.loss_history[k]) + 1), self.loss_history[k], marker='o', linewidth=2)
                axes[i].set_xlabel("Global Round")
                axes[i].set_ylabel("Loss" if k != 'cf_acc' else "Accuracy")
                axes[i].set_title(f"Average {title}")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "client_avg_losses.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ---- 服务端损失 ----
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, v in self.server_loss_history.items():
            ax.plot(range(1, len(v) + 1), v, label=k, marker='o', linewidth=2)
        ax.set_xlabel("Global Round")
        ax.set_ylabel("Loss")
        ax.set_title("Server Training Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "server_losses.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] AE + Server losses saved to {save_dir}")

    def plot_client_losses(self, save_dir):
        """绘制每个客户端的详细损失曲线"""
        client_save_dir = os.path.join(save_dir, "clients")
        os.makedirs(client_save_dir, exist_ok=True)

        for user in self.users:
            if hasattr(user, 'loss_history') and hasattr(user, 'cf_loss_history'):
                # AE损失图
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.flatten()

                ae_loss_keys = ['rec_loss', 'orth_loss', 'align_loss', 'reg_loss']
                ae_titles = ['Reconstruction', 'Orthogonal', 'Alignment', 'Regularization']

                for i, (k, title) in enumerate(zip(ae_loss_keys, ae_titles)):
                    if k in user.loss_history and user.loss_history[k]:
                        axes[i].plot(range(1, len(user.loss_history[k]) + 1),
                                     user.loss_history[k], marker='o', linewidth=2, color='blue')
                        axes[i].set_xlabel("Global Round")
                        axes[i].set_ylabel("Loss")
                        axes[i].set_title(f"Client {user.client_id} - {title} Loss")
                        axes[i].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(client_save_dir, f"client_{user.client_id}_ae_losses.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

                # 分类损失图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                if 'cf_loss' in user.cf_loss_history and user.cf_loss_history['cf_loss']:
                    ax1.plot(range(1, len(user.cf_loss_history['cf_loss']) + 1),
                             user.cf_loss_history['cf_loss'], marker='o', linewidth=2, color='red')
                    ax1.set_xlabel("Global Round")
                    ax1.set_ylabel("Loss")
                    ax1.set_title(f"Client {user.client_id} - Classification Loss")
                    ax1.grid(True, alpha=0.3)

                if 'cf_acc' in user.cf_loss_history and user.cf_loss_history['cf_acc']:
                    ax2.plot(range(1, len(user.cf_loss_history['cf_acc']) + 1),
                             user.cf_loss_history['cf_acc'], marker='o', linewidth=2, color='green')
                    ax2.set_xlabel("Global Round")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_title(f"Client {user.client_id} - Classification Accuracy")
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(client_save_dir, f"client_{user.client_id}_cf_losses.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

        print(f"[INFO] Individual client losses saved to {client_save_dir}")
