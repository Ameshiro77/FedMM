import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.usercream import UserCream
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
from utils.model_utils import make_seq_batch2

class FedCream(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        self.contrast_lr = 1e-3
        self.contrast_steps = 50 
        self.temperature = 0.07

        # 初始化客户端
        self.total_users = len(modalities)
        self.users = []
        
        for i in range(self.total_users):
            client_modals = modalities[i]
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}
            
            client_ae = SplitLSTMAutoEncoder(input_sizes=input_sizes_client, representation_size=rep_size)
            client_cf = MLP(rep_size * len(client_modals), n_classes)
            
            user = UserCream(
                i, self.clients_train_data_list[i], self.test_data, self.public_data, 
                client_ae, client_cf, client_modals, batch_size, learning_rate,
                beta, lamda, local_epochs
            )
            self.users.append(user)
            
        self.server_optimizer = torch.optim.Adam(
            list(self.ae_model_server.parameters()), # CreamFL 主要更新 Encoder
            lr=self.contrast_lr
        )

    def train(self):
        # 预先处理好公共数据的一个 Batch (用于本轮通信)
        # 实际中可以是 DataLoader 循环，这里简化为一个固定 Batch 或每轮随机采样
        modalities_seq, _ = make_seq_batch2(self.public_data, self.batch_size * 2) # 取大一点
        X_pub_all = {m: torch.from_numpy(modalities_seq[m]).float() for m in self.modalities_server}
        seq_len = X_pub_all[self.modalities_server[0]].shape[1]

        # 全局锚点 (Global Anchors)，初始为空
        global_reps = None 
        X_batch_curr = None

        for glob_iter in range(self.num_glob_iters):
            print(f"-------------Round number: {glob_iter} -------------")

            # 1. 准备本轮对比的公共数据片段
            win_len = 32
            idx_start = np.random.randint(0, seq_len - win_len)
            X_batch_curr = {
                m: X_pub_all[m][:, idx_start:idx_start+win_len, :]
                for m in self.modalities_server
            }

            self.selected_users = self.select_users(glob_iter, self.num_users)

            # 2. 客户端本地训练 (传入上一轮生成的 Global Reps)
            for user in self.selected_users:
                # 客户端需要在同样的 X_batch_curr 上做对齐
                # 注意：这里需要把 Tensor 传给客户端，或者客户端自己也有同样的 Public Data Index
                loss = user.train_contrast(global_reps, X_batch_curr)
                print(f"Client {user.client_id} Loss: {loss:.4f}")

            # 3. 服务端聚合 (Ensemble)
            print("Server Aggregating Representations...")
            client_reps_list = []
            for user in self.selected_users:
                # 获取客户端在当前 Public Batch 上的特征
                reps = user.get_public_reps(X_batch_curr)
                if reps is not None:
                    client_reps_list.append(reps)
            
            if len(client_reps_list) > 0:
                # 简单的平均聚合 (Ensemble)
                ensemble_reps = torch.stack(client_reps_list).mean(dim=0).to(self.device)
                
                # 4. 服务端训练 (Server-side Training)
                # 目标：Server Encoder 提取的特征要逼近 Ensemble Reps
                self.train_server_contrast(X_batch_curr, ensemble_reps)
                
                # 5. 更新 Global Reps (用于下一轮下发)
                # 使用更新后的 Server Model 重新计算特征
                self.ae_model_server.eval()
                with torch.no_grad():
                    srv_reps_list = []
                    for m in self.modalities_server:
                        x = X_batch_curr[m].to(self.device)
                        _, out = self.ae_model_server.encoders[m](x)
                        srv_reps_list.append(out[:, -1, :])
                    global_reps = torch.cat(srv_reps_list, dim=-1).detach()
            
            # 测试
            loss, acc, f1 = self.test_server()
            self.rs_glob_acc.append(acc)
            print(f"Server Acc: {acc:.4f}")

        self.save_results()

    def train_server_contrast(self, X_batch, target_reps):
        """
        训练服务端模型以匹配集成的特征 (Ensemble Representations)
        使用 InfoNCE Loss
        """
        self.ae_model_server.train()
        
        for step in range(self.contrast_steps):
            self.server_optimizer.zero_grad()
            
            # Server Forward
            srv_reps_list = []
            for m in self.modalities_server:
                x = X_batch[m].to(self.device)
                _, out = self.ae_model_server.encoders[m](x)
                srv_reps_list.append(out[:, -1, :])
            
            z_server = torch.cat(srv_reps_list, dim=-1)
            
            # Contrastive Loss: Server vs Ensemble
            # z_server [B, D], target_reps [B, D]
            z_server_norm = F.normalize(z_server, dim=1)
            z_target_norm = F.normalize(target_reps, dim=1)
            
            logits = torch.matmul(z_server_norm, z_target_norm.T) / self.temperature
            labels = torch.arange(z_server.size(0)).to(self.device)
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.server_optimizer.step()