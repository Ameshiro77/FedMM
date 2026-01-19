from FLAlgorithms.users.userbase import User
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
import numpy as np
from FLAlgorithms.users.usergao import UserGao
import copy
import torch
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP

class FedGao(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, args, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)
        self.args = args

        # 初始化 Users
        self.total_users = len(modalities)
        for i in range(self.total_users):
            client_modals = modalities[i]
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            # 正常初始化
            client_ae = SplitLSTMAutoEncoder(input_sizes_client, rep_size)
            client_cf = MLP(rep_size, n_classes)

            # 数据分配
            if pfl:
                train_data_user = self.clients_train_data_list[i]
                test_data_user = self.clients_test_data_list[i]
            else:
                train_data_user = self.clients_train_data_list[i]
                test_data_user = self.test_data

            user = UserGao(
                i, train_data_user, test_data_user, self.public_data, client_ae, client_cf,
                client_modals, batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio=label_ratio, args=args)
            
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n-------------Round number: {glob_iter} -------------")

            # 1. 下发最新参数 (只发 AE，因为 Server 基类只有 send_ae_parameters)
            self.send_ae_parameters()
            
            # 注意：分类器参数不分发，客户端沿用自己上一轮的分类器（个性化）
            # 或者如果是普通 FedAvg，会在 aggregate_parameters 里处理
            # 这里我们保持“本地分类器”的设定

            self.selected_users = self.select_users(glob_iter, self.num_users)

            # 2. 客户端无监督 AE 训练
            for user in self.selected_users:
                user.train_ae()

            # 3. 聚合 AE
            self.aggregate_parameters()
            
            # 4. 服务端有监督微调 (只更新 Server 自己的模型，不影响客户端)
            print("[Server] Fine-tuning global model on server labeled data...")
            self.fine_tune_global_model()
            
            # 5. 测试服务端性能
            loss, acc, f1 = self.test_server()
            print(f"[Server Eval] Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)

            # 6. 客户端个性化 (基于本地历史集成)
            # 再次下发刚刚聚合并没有经过 Server 微调的 AE (或者你可以选择下发微调后的)
            # 这里我们下发最新的 Global AE 给客户端做 Teacher 的一部分
            self.send_ae_parameters()
            
            print(f"[Server] Triggering client personalization...")
            for user in self.selected_users:
                # 客户端自己用本地历史模型做 Teacher，不需要 Server 传参
                user.personalize()

    
        accs = self.test_clients()
        self.save_results()

    def fine_tune_global_model(self, epochs=3):
        """ 服务端微调逻辑 """
        self.ae_model_server.train()
        self.cf_model_server.train()
        
        train_data = getattr(self, 'server_train_data', self.public_data)
        if train_data is None or len(train_data['y']) == 0:
            return

        for ep in range(epochs):
            modalities_seq, y_seq = make_seq_batch2(train_data, self.batch_size)
            seq_len = modalities_seq[self.modalities_server[0]].shape[1]
            idx_end = 0
            
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)
                
                self.optimizer_ae.zero_grad()
                self.optimizer_cf.zero_grad()
                
                latents = []
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z = self.ae_model_server.encode(x_m, m)
                    latents.append(z)
                
                logits = self.cf_model_server(torch.cat(latents, dim=-1))
                
                # 逐时间步 Loss
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).long().reshape(-1)
                
                loss = self.cls_loss_fn(logits_flat, y_batch)
                loss.backward()
                self.optimizer_ae.step()
                self.optimizer_cf.step()