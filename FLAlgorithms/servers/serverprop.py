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
            print("client", i, "modals:", client_modals)
            self.users.append(user)

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
            'total_loss': []
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

    def aggregate_and_train(self, z_shares_all_list):
        """
        z_shares_all_list: list, 每个元素是客户端返回的 {mod: [特征]} 字典
        """
        self.cf_model_server.train()
        self.ae_model_server.train()

        z_shares_all = {}  # dict: mod -> list of according client z_share
        for client_dict in z_shares_all_list:
            for mod, z_share in client_dict.items():
                if mod not in z_shares_all:
                    z_shares_all[mod] = []
                z_shares_all[mod].append(z_share)
        # 可视化
        # self._visualize_modal_distributions(z_shares_all)
        # exit()

        # Step 1: 模态内FedAvg
        z_modal_global = []
        for m in z_shares_all.keys():
            z_means = torch.stack([z for z in z_shares_all[m]], dim=0)  # [num_clients_m, D_m]
            z_global_m = z_means.mean(dim=0)  # FedAvg
            z_modal_global.append(z_global_m)
        z_modal_global = torch.stack(z_modal_global, dim=0)  # [num_modal, D]

        # Step 2: 模态间注意力聚合
        attn_scores = torch.softmax(z_modal_global @ self.attn_w.to(self.device), dim=0)  # [num_modal, 1]
        z_global = (attn_scores * z_modal_global).sum(dim=0)  # [D]

        # Step 3: 特征增强
        z_all = torch.cat([torch.cat(z_list, dim=0) for z_list in z_shares_all.values()], dim=0)
        z_aug = torch.cat([z_all, z_global.unsqueeze(0).repeat(z_all.size(0), 1)], dim=1)

        # Step 4: 伪标签生成与增强训练
        pseudo_logits = self.cf_model_server(z_aug.clone().detach())
        pseudo_y = pseudo_logits.argmax(dim=1)

        self.optimizer_cf.zero_grad()
        logits_pred = self.cf_model_server(z_aug)
        pseudo_loss = self.cls_loss_fn(logits_pred, pseudo_y)
        pseudo_loss.backward()
        self.optimizer_cf.step()
        print(f"[Server Enhance] Pseudo loss={pseudo_loss.item():.4f}")

        # Step 5: 有监督训练
        self.train_classifier()

        return z_global

    def train_server(self, z_shares_all_list, glob_iter):
        """
        服务端蒸馏 + 对齐 + 分类联合训练
        """
        self.cf_model_server.train()
        self.ae_model_server.train()

        # === Step 1. 聚合同模态客户端特征 ===
        z_shares_all = {}
        for client_dict in z_shares_all_list:
            for mod, z_share in client_dict.items():
                z_shares_all.setdefault(mod, []).append(z_share)
        
        if glob_iter % 10 == 0:
            self._visualize_modal_distributions(z_shares_all, glob_iter)
            
        z_modal_clients = {}
        for m, zs in z_shares_all.items():
            z_means = torch.stack(zs, dim=0)  # [num_clients_m, D_m]
            z_modal_clients[m] = z_means.mean(dim=0).to(self.device)  # FedAvg

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
                dist_loss, align_loss, cls_loss = 0.0, 0.0, 0.0

                # === Step 3. 计算每个模态的server特征 & 蒸馏 ===
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z_m = self.ae_model_server.encode(x_m, m)
                    z_m_all[m] = z_m

                    # 对服务器序列特征进行池化，变成静态特征
                    z_m_pooled = z_m.mean(dim=1)  # [batch_size, feature_dim] - 平均池化

                    dist_loss += torch.nn.functional.mse_loss(
                        z_m_pooled,
                        z_modal_clients[m].unsqueeze(0).expand(self.batch_size, -1).detach()
                    )

                # === Step 4. 模态间语义对齐 ===
                z_stack = torch.stack(list(z_m_all.values()), dim=0)  # [M, B,win, rep]\

                z_center = z_stack.mean(dim=0)
                for m in z_m_all.keys():
                    align_loss += torch.nn.functional.mse_loss(z_m_all[m], z_center)

                # === Step 5. 分类训练 ===
                z_fuse = torch.cat([z for z in z_m_all.values()], dim=-1)
                y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device).flatten().long()

                logits = self.cf_model_server(z_fuse)
                # print(logits.shape,y_true.shape)
                cls_loss = self.cls_loss_fn(logits, y_true)

                # === Step 6. 总损失 ===
                total_loss = (
                    0.1 * dist_loss +
                    0.1 * align_loss +
                    1 * cls_loss
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

        print(z_global_center.shape, "111")
        return z_global_center.mean(dim=0)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- Global Round [{glob_iter+1}/{self.num_glob_iters}] ---")

            # Step 1: 客户端提取 z_share
            self.selected_users = self.select_users(glob_iter, self.num_users)

            # 现在改为字典形式存储所有模态

            z_shares_all = []
            for user in self.selected_users:
                # 每个客户端返回的是 { mod1: [z_share1, z_share2, ...], mod2:.. }
                z_shares_client = user.upload_shares()
                z_shares_all.append(z_shares_client)

            # Step 2: 服务器聚合 + 伪标签增强训练
            # z_global = self.aggregate_and_train(z_shares_all)
            z_global = self.train_server(z_shares_all, glob_iter)

            # Step 3: 客户端更新 AE
            epoch_losses_clients = {'rec_loss': [], 'orth_loss': [], 'align_loss': [], 'reg_loss': []}
            for user in self.selected_users:
                pre_w = [p.clone().detach().cpu() for p in user.ae_model.parameters()]
                z_shares_new, loss_dict = user.train_ae_prop(z_global, pre_w)
                print(f"[Client {user.client_id}] Recon={loss_dict['rec_loss']:.4f}")

                # 累加各客户端本轮损失
                for k in epoch_losses_clients.keys():
                    epoch_losses_clients[k].append(loss_dict[k])

            # Step 4: 记录平均损失
            for k in self.loss_history.keys():
                avg_loss = np.mean(epoch_losses_clients[k])
                self.loss_history[k].append(avg_loss)

            # Step 5: 客户端分类器训练
            for user in self.users:
                user.train_cf_prop()

            # Step 6: 服务端测试
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)

        # Step 7: 绘图与保存
        self.plot_losses(save_dir=f"./results/{self.dataset}/")
        self.test_clients()
        self.save_results()

    def plot_losses(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # ---- 客户端损失 ----
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, k in enumerate(self.loss_history.keys()):
            axes[i].plot(range(1, len(self.loss_history[k]) + 1), self.loss_history[k], marker='o')
            axes[i].set_xlabel("Global Round")
            axes[i].set_ylabel("Loss")
            axes[i].set_title(f"Client {k}")
            axes[i].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "client_losses.png"))
        plt.close()

        # ---- 服务端损失 ----
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, v in self.server_loss_history.items():
            ax.plot(range(1, len(v) + 1), v, label=k, marker='o')
        ax.set_xlabel("Global Round")
        ax.set_ylabel("Loss")
        ax.set_title("Server Training Losses")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "server_losses.png"))
        plt.close()

        print(f"[INFO] AE + Server losses saved to {save_dir}")
