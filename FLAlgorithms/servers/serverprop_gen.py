from FLAlgorithms.users.userbase import User
# from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.users.userprop_gen import *
import json
import codecs
from FLAlgorithms.trainmodel.ae_model import *
from FLAlgorithms.trainmodel.losses import *
from torch import nn, optim
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class FedPropGen(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times)

        self.total_users = len(modalities)

        # self.cf_model_server = MLP(rep_size, n_classes).to(self.device)

        self.attn_temperature = 1.0  # 可调
        self.n_classes = n_classes 
        self.generator = StyleAwareGenerator(
            style_dim=rep_size, 
            n_classes=n_classes, 
            hidden_dim=128, 
            content_dim=rep_size
        ).to(self.device)
        
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=0.001)
        # ===== (A) Add learnable modality weights =====
        self.modality_weights = torch.nn.Parameter(
            torch.zeros(len(self.modalities_server)),  # one weight per modality
            requires_grad=True
        )
        self.modality_weight = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in self.modalities_server
        })

        self.softmax = torch.nn.Softmax(dim=0)

        self.attn_w = nn.Parameter(torch.randn(rep_size, 1).to(self.device) * 0.1).to(self.device)
        self.proj_heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(rep_size, rep_size), nn.ReLU(), nn.Linear(rep_size, rep_size))
            for m in self.modalities_server
        }).to(self.device)
        nn.init.xavier_uniform_(self.attn_w)
        self.cf_model_server = MLP(rep_size, n_classes).to(self.device)
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
            user = UserPropGen(
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
        2D 可视化每个客户端的 prototype 分布。
        每个点 = 某客户端的某一类原型
        颜色 = 模态类型
        """
        all_feats, all_modalities, all_labels = [], [], []

        for client_idx, (proto_dict, modality) in enumerate(prototypes_weights):
            for cls, proto in proto_dict.items():
                z_np = proto.detach().cpu().numpy().reshape(-1)
                all_feats.append(z_np)
                all_modalities.append(modality)
                all_labels.append(int(cls))

        all_feats = np.stack(all_feats)

        # === t-SNE 降维到2D ===
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        z_emb = tsne.fit_transform(all_feats)

        # === 绘制2D图 ===
        plt.figure(figsize=(10, 8))

        # 获取唯一模态并分配颜色
        unique_modalities = sorted(set(all_modalities))
        palette = sns.color_palette("tab10", len(unique_modalities))

        # 为每个模态绘制散点
        for i, modality in enumerate(unique_modalities):
            mask = [m == modality for m in all_modalities]
            modality_emb = z_emb[mask]
            modality_labels = np.array(all_labels)[mask]

            plt.scatter(
                modality_emb[:, 0], modality_emb[:, 1],
                color=palette[i],
                label=f'Modality {modality}',
                s=60, alpha=0.8, edgecolor='k', linewidth=0.2
            )

            # 可选：在点上标注类别标签
            for j, (x, y) in enumerate(modality_emb):
                plt.annotate(str(modality_labels[j]), (x, y),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=6, alpha=0.7)

        plt.title(f"2D Prototype Distribution by Modality (Iter {glob_iter})", fontsize=14)
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend(title="Modalities")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/dist/prototypes2D_{self.dataset}_{glob_iter}.png", dpi=300)
        plt.close()

    def train_server(self, z_shares_all_list, z_specs_stat_list, prototypes_weights, glob_iter):
            """
            服务端蒸馏 + 对齐 + 分类联合训练 + 生成器增强
            """
            self.cf_model_server.train()
            self.ae_model_server.train()
            self.generator.train() # [新增] 开启生成器训练

           
            # === Step 0. 预处理  ===

            # 结构: style_bank[modality] = list of (mean, std)
            style_bank = {}
            for client_stat in z_specs_stat_list:
                for m, stat in client_stat.items():
                    if m in self.modalities_server:
                        style_bank.setdefault(m, []).append(
                            (stat['mean'].to(self.device), stat['std'].to(self.device))
                        )

            real_features_list = []
            for client_dict in z_shares_all_list:
                for m, z in client_dict.items():
                    if m in self.modalities_server:
                        real_features_list.append(z.to(self.device))
            
            real_features = None
            if len(real_features_list) > 0:
                real_features = torch.cat(real_features_list, dim=0) # [Total_Samples, D]


            # === Step 1. 聚合同模态客户端特征 (FedAvg / Mean pooling) ===
            z_shares_all = {}
            for client_dict in z_shares_all_list:
                for mod, z_share in client_dict.items():
                    z_shares_all.setdefault(mod, []).append(z_share)

            if glob_iter % 10 == 0:
                pass
                # self._visualize_modal_distributions(prototypes_weights, glob_iter)

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
                    loss_gen_total = 0.0 # 生成器Loss

                    # === Step 3A. 计算服务端特征 & 蒸馏 & 模态间对齐
                    for m in self.modalities_server:
                        x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                        _, z_m = self.ae_model_server.encode(x_m, m)      # [B, T, D]
                        z_m_all[m] = z_m
                        z_m_pooled = z_m.mean(dim=1)                      # [B, D]

                        # ====== 准备客户端特征 bank (用于蒸馏) ======
                        z_s = F.normalize(z_m_pooled, dim=-1)          # [B, D]
                        if m in z_shares_all and len(z_shares_all[m]) > 0:
                            z_c = F.normalize(torch.cat(z_shares_all[m], dim=0).to(self.device), dim=-1)  # [N_total, D]
                            
                            # ====== MMD 蒸馏损失 ======
                            mmd_loss_val = compute_mmd(z_s, z_c)
                            dist_loss += mmd_loss_val

                    # === 模态间语义对齐 (Contrastive Alignment) ===
                    z_proj_dict = {m: self.proj_heads[m](z_m_all[m]) for m in z_m_all.keys()}
                    align_loss = contrastive_modality_align(z_proj_dict)

                    # === Step 3B. 训练生成器 
                    if real_features is not None and len(style_bank) > 0:
                        self.optimizer_gen.zero_grad()
                        
                        # A. 采样 Style (重参数化)
                        m_sel = np.random.choice(list(style_bank.keys()))
                        stats_list = style_bank[m_sel]
                        
                        # 确保 stats_list 不为空
                        if len(stats_list) > 0:
                            mu, std = stats_list[np.random.randint(len(stats_list))]
                            
                            # 生成正态噪声并通过 mu/std 变换为特定风格
                            # noise shape: [Batch, Style_Dim]
                            noise = torch.randn(self.batch_size, mu.shape[0]).to(self.device)
                            z_style_fake = mu + noise * std 
                            
                            # B. 生成伪标签和伪特征
                            y_gen = torch.randint(0, self.n_classes, (self.batch_size,)).to(self.device)
                            z_share_gen = self.generator(z_style_fake, y_gen)
                            
                            # C. Loss 1: 语义一致性 (让分类器能识别它) - 此时不更新分类器，只用作判别
                            # 注意：z_share_gen 是特征，直接送入 MLP
                            logits_gen = self.cf_model_server(z_share_gen)
                            loss_sem = self.cls_loss_fn(logits_gen, y_gen)
                            
                            # D. Loss 2: 分布匹配 (MMD) - 让生成的 z_share 像真的客户端 z_share
                            idx_real = torch.randperm(real_features.size(0))[:self.batch_size]
                            z_real_batch = real_features[idx_real]
                            
                            # 确保维度匹配
                            if z_real_batch.size(0) == self.batch_size:
                                loss_mmd_gen = compute_mmd(z_share_gen, z_real_batch)
                            else:
                                loss_mmd_gen = torch.tensor(0.0).to(self.device)
                            
                            # E. 反向传播更新生成器
                            loss_gen_total = loss_sem + 0.5 * loss_mmd_gen
                            loss_gen_total.backward()
                            self.optimizer_gen.step()


                    # === Step 4. 分类器训练 (真实数据 + 生成数据增强)
                    # 1. 准备真实数据融合特征
                    z_fuse = self.fusionNet(z_m_all)  # B,W,D
                    z_fuse_all.append(z_fuse)
                    y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device).flatten().long()

                    # 2. 计算真实数据 Loss
                    logits_real = self.cf_model_server(z_fuse)
                    loss_cls_real = self.cls_loss_fn(logits_real, y_true)

                    # 3. 计算生成数据增强 Loss (Data Augmentation)
                    loss_cls_fake = 0.0
                    if len(style_bank) > 0:
                        with torch.no_grad(): # 生成过程不求导
                            m_sel = np.random.choice(list(style_bank.keys()))
                            stats_list = style_bank[m_sel]
                            if len(stats_list) > 0:
                                mu, std = stats_list[np.random.randint(len(stats_list))]
                                noise = torch.randn(self.batch_size, mu.shape[0]).to(self.device)
                                z_style_aug = mu + noise * std
                                y_aug = torch.randint(0, self.n_classes, (self.batch_size,)).to(self.device)
                                
                                z_share_aug = self.generator(z_style_aug, y_aug)
                        
                        if 'z_share_aug' in locals():
                            logits_aug = self.cf_model_server(z_share_aug)
                            loss_cls_fake = self.cls_loss_fn(logits_aug, y_aug)

                    # 4. 合并分类 Loss
                    cls_loss = loss_cls_real + 0.5 * loss_cls_fake

                    # 更新 Logits Bank (仅使用真实数据的结果)
                    for cls_id in y_true.unique():
                        cls_mask = (y_true == cls_id)
                        if cls_mask.sum() > 0:
                            logits_cls = logits_real[cls_mask].mean(dim=0)
                            if not hasattr(self, "logits_bank"):
                                self.logits_bank = {}
                            if cls_id.item() not in self.logits_bank:
                                self.logits_bank[cls_id.item()] = logits_cls.clone().detach()
                            else:
                                self.logits_bank[cls_id.item()] = (
                                    0.5 * self.logits_bank[cls_id.item()] + 0.5 * logits_cls.clone().detach()
                                )

                    # =======================================================================
                    # === Step 5. 总损失反向传播 ===
                    # =======================================================================
                    total_loss = (
                        0.5 * dist_loss +
                        0.5 * align_loss +
                        1.0 * cls_loss
                    )

                    total_loss.backward()
                    self.optimizer_ae.step()
                    self.optimizer_cf.step()

                print(f"[Server Train] Distill={dist_loss:.4f} Align={align_loss:.4f} Cls_Real={loss_cls_real:.4f} Cls_Fake={loss_cls_fake:.4f} Gen_Loss={loss_gen_total:.4f}")

            # 记录 Loss
            self.server_loss_history['distill_loss'].append(dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss)
            self.server_loss_history['cls_loss'].append(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss)
            self.server_loss_history['total_loss'].append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)

            return torch.cat(z_fuse_all, dim=1).mean(dim=1).detach(), None, self.logits_bank

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- Global Round [{glob_iter+1}/{self.num_glob_iters}] ---")

            # Step 1: 客户端提取 z_share
            self.selected_users = self.select_users(glob_iter, self.num_users)
            z_shares_all = [user.upload_shares() for user in self.selected_users]
            z_specs_stat_all = [user.upload_spec() for user in self.selected_users]
            prototypes_weights = [(user.upload_prototype(), user.get_prototype_weight())
                                  for user in self.selected_users]

            # Step 2: 服务器训练
            z_global, global_prototypes, server_logits = self.train_server(z_shares_all, z_specs_stat_all, prototypes_weights, glob_iter)

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

            # uplink_bytes = 0
            # for client_dict in z_shares_all:
            #     for z_share in client_dict.values():
            #         uplink_bytes += z_share.numel() * 4
            # for proto_dict, weight_dict in prototypes_weights:
            #     for proto in proto_dict.values():
            #         uplink_bytes += proto.numel() * 4
            #     uplink_bytes += len(weight_dict) * 4

            # # 下行通信量 (发送给每个选中的客户端)
            # downlink_bytes = 0
            # downlink_bytes += z_global.numel() * 4 * len(self.selected_users)  # z_global
            # for logits in server_logits.values():
            #     downlink_bytes += logits.numel() * 4 * len(self.selected_users)  # server_logits

            # print(f"\n=== 单轮通信量统计 ===")
            # print(f"选中客户端数量: {len(self.selected_users)}")
            # print(f"单轮上行通信量: {uplink_bytes/1024:.2f} KB")
            # print(f"单轮下行通信量: {downlink_bytes/1024:.2f} KB")
            # print(f"单轮总通信量: {(uplink_bytes+downlink_bytes)/1024:.2f} KB")
            # exit()

            # Step 6: 测试
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)

        save_path = f"./results/{self.dataset}/loss_server.json"
        client_loss_all = {}
        for user in self.selected_users:
            client_loss_all[user.client_id] = {
                "ae_loss": user.loss_history,
                "cf_loss": user.cf_loss_history
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


    def test_server(self):
        """服务器端测试集评估"""
        self.ae_model_server.eval()
        self.cf_model_server.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        eval_win = EVAL_WIN
        # 添加F1 Score计算所需的变量
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
                    # hidden_seq: [1, T, D]
                    z_m_all[m] = hidden_seq    # 直接保持与训练阶段一致

      
                z_fuse = self.fusionNet(z_m_all)
                outputs = self.cf_model_server(z_fuse)

                loss = self.cls_loss_fn(outputs, batch_y)
                preds = torch.argmax(outputs, dim=1)

                total_loss += loss.item() * batch_y.size(0)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
                
                # 收集预测结果和真实标签
                all_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())


        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')  # 对于多分类使用加权平均
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        print(f"Server Test - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, avg_acc, f1
