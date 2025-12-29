import torch
import torch.nn as nn
import numpy as np
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.usergao import UserGao
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
from utils.model_utils import make_seq_batch2

class FedGao(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        # 服务端有标签数据的优化器 (用于 Phase 2)
        self.server_optimizer = torch.optim.Adam(
            list(self.ae_model_server.parameters()) + list(self.cf_model_server.parameters()),
            lr=0.001
        )
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # 初始化客户端
        self.total_users = len(modalities)
        self.users = []
        
        for i in range(self.total_users):
            client_modals = modalities[i]
            
            # 统一模态处理逻辑
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
                num_modals = len(client_modals)
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}
                num_modals = 1 # 单模态
            
            client_ae = SplitLSTMAutoEncoder(input_sizes=input_sizes_client, representation_size=rep_size)
            client_cf = MLP(rep_size * num_modals, n_classes)
            if pfl:
                user = UserGao(
                i, self.clients_train_data_list[i], self.clients_test_data_list[i], self.public_data, 
                client_ae, client_cf, client_modals, batch_size, learning_rate,
                beta, lamda, local_epochs, label_ratio=label_ratio
            )
            else:
                user = UserGao(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data, 
                    client_ae, client_cf, client_modals, batch_size, learning_rate,
                    beta, lamda, local_epochs
                )
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"-------------Round number: {glob_iter} -------------")
            
            self.selected_users = self.select_users(glob_iter, self.num_users)

            # === Phase 1: Client Unsupervised Representation Learning ===
            # 客户端利用本地无标签数据训练 AE
            for user in self.selected_users:
                # 1. 下发全局 AE 参数
                user.ae_model.load_state_dict(self.ae_model_server.state_dict())
                
                # 2. 本地无监督训练
                loss_ae = user.train_ae()
                print(f"Client {user.client_id} AE Loss: {loss_ae:.4f}")
            
            # 3. 聚合 AE 参数 (使用 ServerBase 的标准聚合方法)
            # 注意：这里只会聚合 ae_model_server，因为 user 只更新了 ae_model
            # 假设 self.aggregate_parameters() 会遍历 selected_users 并平均 ae_model
            self.aggregate_parameters() 

            # === Phase 2: Server Supervised Fine-tuning ===
            # 服务端利用少量有标签数据 (self.server_train_data)
            # 对 AE + Classifier 进行联合训练
            print("Server executing Supervised Fine-tuning...")
            server_loss, server_acc = self.train_server_supervised()
            print(f"Server Supervised Loss: {server_loss:.4f}, Acc: {server_acc:.4f}")
            
            # === Phase 3: Client Personalization (Pseudo-labeling) ===
            # 将训练好的全局模型 (AE + CF) 发送给所有客户端，进行个性化微调
            # 这一步主要是为了评估个性化性能，或者为下一轮提供更好的伪标签基础
            
            avg_p_loss = 0
            for user in self.users: 
                # 1. 接收完整的全局模型 (AE + CF)
                user.ae_model.load_state_dict(self.ae_model_server.state_dict())
                user.cf_model.load_state_dict(self.cf_model_server.state_dict())
                
                # 2. 本地基于伪标签微调
                loss_p = user.train_personalization()
                avg_p_loss += loss_p
            
            print(f"Avg Personalization Loss: {avg_p_loss / len(self.users):.4f}")

            # === Test ===
            # 此时测试的是客户端经过个性化微调后的模型
            loss, acc, f1 = self.test_server() # 这里通常测试的是 Server 模型在 TestSet 上的表现
            # 如果要测试客户端个性化后的表现：
            accs = self.test_clients() 
            avg_acc = np.mean(accs)
            
            print(f"Global Round {glob_iter}, Server Test Acc: {acc:.4f}, Avg Client Acc: {avg_acc:.4f}")
            self.rs_glob_acc.append(avg_acc)

        self.save_results()

    def train_server_supervised(self):
        """
        [完整实现] 服务端有监督微调：
        利用 server_train_data 训练 ae_model_server 和 cf_model_server
        """
        self.ae_model_server.train()
        self.cf_model_server.train()
        
        # 1. 准备服务端数据
        # 假设 self.server_train_data 包含所有模态数据和标签
        modalities_seq, y_seq = make_seq_batch2(self.server_train_data, self.batch_size)
        X_server = {m: modalities_seq[m] for m in self.modalities_server}
        seq_len = X_server[self.modalities_server[0]].shape[1]
        y_batch_full = torch.from_numpy(y_seq).long().to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 服务端训练 epoch (例如 5 轮)
        for epoch in range(5):
            idx_end = 0
            while idx_end < seq_len:
                win_len = 32 # 固定窗口或随机
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                self.server_optimizer.zero_grad()
                
                # Forward
                reps_list = []
                for m in self.modalities_server:
                    # 确保数据转换为 Tensor
                    x_np = X_server[m][:, idx_start:idx_end, :]
                    x_seg = torch.from_numpy(x_np).float().to(self.device)
                    
                    # 提取特征
                    _, out = self.ae_model_server.encoders[m](x_seg)
                    reps_list.append(out[:, -1, :])
                
                # 拼接并分类
                if len(reps_list) > 0:
                    rep_cat = torch.cat(reps_list, dim=-1)
                    logits = self.cf_model_server(rep_cat)
                    
                    # 标签
                    y_seg = y_batch_full[:, idx_start].reshape(-1)
                    
                    # 计算损失
                    loss = self.cls_loss_fn(logits, y_seg)
                    loss.backward()
                    self.server_optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 计算准确率
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_seg).sum().item()
                    total_samples += y_seg.size(0)
        
        avg_loss = total_loss / (total_samples / self.batch_size * 5 + 1e-6) # 估算 batch 数
        avg_acc = total_correct / (total_samples + 1e-6)
        
        return avg_loss, avg_acc