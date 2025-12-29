import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.userdf import UserDF # 引用上面的 UserDF
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
from utils.model_utils import make_seq_batch2

class FedDF(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        # FedDF 超参数
        self.temp = 1.0     # 蒸馏温度
        self.distill_lr = 1e-3
        self.distill_steps = 100 # 每一轮服务端的蒸馏步数 [cite: 307]
        
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
            client_cf = MLP(rep_size * len(client_modals), n_classes) # 假设简单的 concat
            if pfl:
                user = UserDF(
                i, self.clients_train_data_list[i], self.clients_test_data_list[i], self.public_data, 
                client_ae, client_cf, client_modals, batch_size, learning_rate,
                beta, lamda, local_epochs
            )
            else:
                user = UserDF(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data, 
                    client_ae, client_cf, client_modals, batch_size, learning_rate,
                    beta, lamda, local_epochs
                )
            self.users.append(user)
            
        # 服务端优化器 (针对全局模型)
        self.server_optimizer = torch.optim.Adam(
            list(self.ae_model_server.parameters()) + list(self.cf_model_server.parameters()),
            lr=self.distill_lr
        )

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"-------------Round number: {glob_iter} -------------")

            # 1. 选取客户端
            self.selected_users = self.select_users(glob_iter, self.num_users)

            # 2. 客户端本地训练 [cite: 66]
            for user in self.selected_users:
                # 同步全局模型参数到客户端 (这是 FedDF 的标准步骤，客户端从融合后的模型开始) 
                # 注意：这里需要处理模态异构，如果完全异构则不能直接 load_state_dict
                # 为简化，假设 UserDF.train_local 会基于当前参数继续训练
                loss = user.train_local()
                print(f"Client {user.client_id} Local Loss: {loss:.4f}")

            # 3. 服务端集成蒸馏 (Ensemble Distillation) [cite: 10, 70]
            self.ensemble_distillation()

            # 4. (可选) 将蒸馏后的模型分发给客户端
            # 在下一轮开始时，客户端通常会下载这个模型。
            # 或者是像 FedAvg 一样，先聚合参数再蒸馏。
            # 原文 FedDF 流程：Client Update -> Server Aggregation (Optional) -> Server Distillation -> Broadcast [cite: 69]
            # 这里我们简化为：Client Update -> Server Distillation (using Client Models as Teachers)
            
            # 测试
            loss, acc, f1 = self.test_server()
            print(f"Server Distill Test Acc: {acc:.4f}")
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)

        self.save_results()

    def ensemble_distillation(self):
        """
        FedDF 的核心：在服务端利用公共数据进行 Logits 级蒸馏 [cite: 72, 73]
        """
        print("Server executes Ensemble Distillation...")
        self.ae_model_server.train()
        self.cf_model_server.train()
        
        # 准备公共数据
        modalities_seq, _ = make_seq_batch2(self.public_data, self.batch_size)
        X_pub = {m: modalities_seq[m] for m in self.modalities_server}
        seq_len = X_pub[self.modalities_server[0]].shape[1]
        
        # 蒸馏循环
        for step in range(self.distill_steps):
            # 随机采样一个窗口
            win_len = 32
            if seq_len > win_len:
                idx_start = np.random.randint(0, seq_len - win_len)
            else:
                idx_start = 0
            idx_end = idx_start + win_len
            
            # 构造 Batch
            X_batch = {
                m: torch.from_numpy(X_pub[m][:, idx_start:idx_end, :]).float().to(self.device)
                for m in self.modalities_server
            }

            self.server_optimizer.zero_grad()
            
            # --- Step 3.1: 获取教师集成的 Logits (AvgLogits) ---
            teacher_logits_list = []
            
            for user in self.selected_users:
                # 让每个客户端对公共数据进行推理 [cite: 74]
                # 注意：需要处理模态缺失，如果客户端没有该模态的数据，可能需要跳过或补零
                # 这里假设 user.get_logits 内部处理了模态匹配
                logits = user.get_logits(X_batch) 
                if logits is not None:
                    teacher_logits_list.append(logits)
            
            if len(teacher_logits_list) == 0:
                continue

            # 堆叠并计算平均 Logits (Ensembling) [cite: 18, 76]
            teacher_logits_stack = torch.stack(teacher_logits_list)
            # Softmax 后平均，或者 Logits 平均，FedDF 论文公式使用的是 Logits 平均后 Softmax [cite: 76]
            # AvgLogits: mean(f(x))
            avg_teacher_logits = torch.mean(teacher_logits_stack, dim=0)
            
            # 计算教师的软标签分布 (Soft Targets)
            teacher_probs = F.softmax(avg_teacher_logits / self.temp, dim=1)

            # --- Step 3.2: 获取学生 (Server) 的 Logits ---
            # Server 前向传播
            reps = []
            for m in self.modalities_server:
                _, out = self.ae_model_server.encoders[m](X_batch[m])
                reps.append(out[:, -1, :])
            rep_cat = torch.cat(reps, dim=-1)
            student_logits = self.cf_model_server(rep_cat)
            
            # --- Step 3.3: 计算 KL 散度损失 ---
            # Loss = KL(Student || Teacher) [cite: 76]
            student_log_probs = F.log_softmax(student_logits / self.temp, dim=1)
            
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temp ** 2)
            
            loss.backward()
            self.server_optimizer.step()
            
            if step % 20 == 0:
                print(f"  Distill Step {step}, KL Loss: {loss.item():.4f}")