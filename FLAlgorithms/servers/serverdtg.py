from FLAlgorithms.users.userbase import User
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
import numpy as np
from FLAlgorithms.users.userdtg import UserDTG
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP,StyleAwareGenerator
from torch import nn, optim
from torch.nn import functional as F
import torch

class FedDTG(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        self.n_classes = n_classes
        self.rep_size = rep_size
        self.total_users = len(modalities)

        # === 1. 初始化生成器 (Baseline 核心) ===
        self.generator = StyleAwareGenerator(
            style_dim=rep_size, 
            n_classes=n_classes, 
            hidden_dim=128, 
            content_dim=rep_size
        ).to(self.device)
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=0.001)

        # === 2. 初始化 Server 分类器 ===
        # 【严格遵照指令】使用 rep_size

        # === 3. 初始化 Client ===
        for i in range(self.total_users):
            client_modals = modalities[i]
            
            # 处理 input_sizes
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size
            )
            

            client_cf = MLP(rep_size, n_classes) 

            if pfl:
                user = UserDTG(
                i, self.clients_train_data_list[i], self.clients_test_data_list[i], self.public_data, client_ae, client_cf,
                client_modals, batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio=label_ratio)
            else:
                user = UserDTG(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data, client_ae, client_cf,
                    client_modals, batch_size, learning_rate, beta, lamda,
                    local_epochs, label_ratio=label_ratio)
            self.users.append(user)

    def train_server(self, z_specs_stat_list):
        self.generator.train()
        self.cf_model_server.train()
        self.ae_model_server.train()

        # 1. 構建 Style Bank
        style_bank = {}
        for client_stat in z_specs_stat_list:
            for m, stat in client_stat.items():
                if m in self.modalities_server:
                    style_bank.setdefault(m, []).append(
                        (stat['mean'].to(self.device), stat['std'].to(self.device))
                    )

        if not style_bank: return

        # 2. 準備服務端真實數據
        modalities_seq, labels = make_seq_batch2(self.server_train_data, self.batch_size)
        seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]

        # 訓練循環
        for epoch in range(1): # 可根據需要調整 epoch
            idx_end = 0
            while idx_end < seq_len_batch:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len_batch)

                self.optimizer_gen.zero_grad()
                self.optimizer_cf.zero_grad()

                # ==========================================
                # Part A: 處理真實數據 (Real Data) - 拼接
                # ==========================================
                z_real_list = []
                
                # 對每個模態編碼 -> 存入列表
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    _, z_m_seq = self.ae_model_server.encode(x_m, m)
                    
                    # 取均值代表該窗口特徵
                    z_m_mean = z_m_seq.mean(dim=1) 
                    z_real_list.append(z_m_mean)

                # 【關鍵】拼接真實特徵 [Batch, Rep_Size * Num_Modals]
                z_real_cat = torch.cat(z_real_list, dim=1) 
                
                y_true = torch.from_numpy(labels[:, idx_start:idx_end]).to(self.device)
                y_true = y_true[:, 0].long() # 取窗口第一個標籤

                # 計算 Real Loss
                logits_real = self.cf_model_server(z_real_cat)
                loss_cls_real = self.cls_loss_fn(logits_real, y_true)

                # ==========================================
                # Part B: 處理生成數據 (Fake Data) - 分模態生成後拼接
                # ==========================================
                # 生成一批統一的標籤 (假設所有模態共享這個標籤)
                y_fake = torch.randint(0, self.n_classes, (self.batch_size,)).to(self.device)
                z_fake_list = []

                # 【關鍵】針對每個模態分別生成
                for m in self.modalities_server:
                    # 從該模態的 Bank 中採樣風格
                    if m in style_bank and len(style_bank[m]) > 0:
                        stats_list = style_bank[m]
                        mu, std = stats_list[np.random.randint(len(stats_list))]
                        
                        noise = torch.randn(self.batch_size, mu.shape[0]).to(self.device)
                        z_style = mu + noise * std
                        
                        # 生成該模態的特徵
                        z_fake_m = self.generator(z_style, y_fake)
                        z_fake_list.append(z_fake_m)
                    else:
                        # 兜底：如果 bank 裡沒有這個模態，生成全 0 防止報錯
                        z_fake_list.append(torch.zeros(self.batch_size, self.rep_size).to(self.device))

                # 【關鍵】拼接生成特徵 [Batch, Rep_Size * Num_Modals]
                z_fake_cat = torch.cat(z_fake_list, dim=1)

                # 1. 計算生成器的 Class Loss (讓生成的特徵能被分類器識別)
                logits_fake = self.cf_model_server(z_fake_cat)
                loss_gen_ce = self.cls_loss_fn(logits_fake, y_fake)

                # 2. 計算特徵匹配 Loss (讓生成的 cat 特徵分佈接近真實的 cat 特徵)
                idx_real = torch.randperm(z_real_cat.size(0))[:self.batch_size]
                z_real_batch = z_real_cat[idx_real]
                
                # 確保維度一致 (處理最後一個 batch 可能不足的情況)
                if z_real_batch.size(0) == z_fake_cat.size(0):
                    loss_gen_match = F.mse_loss(z_fake_cat, z_real_batch)
                else:
                    loss_gen_match = torch.tensor(0.0).to(self.device)

                # Generator Update
                loss_gen_total = loss_gen_ce + 0.5 * loss_gen_match
                loss_gen_total.backward(retain_graph=True) # 保留圖給 Classifier 用
                self.optimizer_gen.step()

                # ==========================================
                # Part C: 訓練分類器 (Classifier Update)
                # ==========================================
                # 使用 Detach 後的 fake data 增強分類器
                logits_fake_eval = self.cf_model_server(z_fake_cat.detach())
                loss_cls_fake = self.cls_loss_fn(logits_fake_eval, y_fake)

                loss_cf_total = loss_cls_real + 0.5 * loss_cls_fake
                loss_cf_total.backward()
                self.optimizer_cf.step()

                print(f"[FedDTG-Server] Gen={loss_gen_total:.4f} Real={loss_cls_real:.4f} Fake={loss_cls_fake:.4f}")

        return None



    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"--- Round {glob_iter} ---")
            
            self.send_ae_parameters() 
            self.selected_users = self.select_users(glob_iter, self.num_users)

            z_specs_list = []

            for user in self.selected_users:
                # Client 本地训练
                user.train_ae()
                user.train_cf()
                
                # 只上传统计量
                z_specs_list.append(user.upload_spec())

            # Server 利用统计量训练生成器
            self.train_server(z_specs_list)

            # 聚合参数
            self.aggregate_parameters()
            
            # Test
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            print(f"Server Acc: {acc:.4f}")
            self.rs_glob_f1.append(f1)
         
        print("all saved2")
        accs = self.test_clients()
        print("Test accuracy: ", accs)
        print("all saved")
        self.save_results()