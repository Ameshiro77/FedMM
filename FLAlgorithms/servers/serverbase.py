import torch
import os
import numpy as np
import h5py
import copy
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils import *
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP, FusionNet, DynamicGatedFusion
import matplotlib.pyplot as plt
from FLAlgorithms.config import EVAL_WIN


class Server:
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, pfl):

        self.dataset = dataset

        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.algorithm = algorithm
        self.beta = beta
        self.lamda = lamda
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_glob_f1 = []
        self.times = times
        self.users = []
        self.selected_users = []
        self.num_users = num_users

        self.total_users = len(modalities)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create / copy models
        # ae_model_class_or_dict may be a class or a dict of modules
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.fusionNet = FusionNet(modalities, rep_size).to(self.device)
        self.fusionNet = DynamicGatedFusion(modalities, rep_size).to(device)

        self.rec_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

        train_data, test_data, public_data = load_data(dataset)
        self.public_data = split_public(public_data, dataset=dataset, ratio=0.5)
        self.test_data = test_data

        # 服务端的train_data全是labeled
        self.server_train_data, self.clients_train_data = split_server_train(train_data, dataset=dataset, ratio=0.1)

        self.pfl = pfl
        if pfl:
            self.clients_train_data_list, self.clients_test_data_list = split_clients_train_and_test(
                self.clients_train_data,
                self.total_users,
                self.dataset,
                test_ratio=0.25
            )
        else:
            self.clients_train_data_list = split_clients_train(self.clients_train_data, self.total_users, self.dataset)

        self.modalities_server = [m for m in self.server_train_data if m != "y"]
        self.ae_model_server = SplitLSTMAutoEncoder(input_sizes, rep_size).to(device)
        self.cf_model_server = MLP(rep_size*len(self.modalities_server), n_classes).to(device)
        # self.cf_model_server = MLP(rep_size, n_classes).to(device)

        self.optimizer_ae = torch.optim.Adam(self.ae_model_server.parameters(), lr=learning_rate)
        self.optimizer_cf = torch.optim.Adam(self.cf_model_server.parameters(), lr=learning_rate)

        print("samples in server train:", len(self.server_train_data['y']))
        print("samples in clients train total:", len(self.clients_train_data['y']))
        print("actual public data samples:", len(self.public_data['y']))
        for idx, data in enumerate(self.clients_train_data_list):
            print("client:", idx, ";samples:", len(data['y']))
        print_dataset_info(self.server_train_data, name="Server Train")
        print_dataset_info(self.clients_train_data, name="Clients Train")
        print_dataset_info(self.public_data, name="Public")
        print_dataset_info(self.test_data, name="Test")

        # user逻辑须在子类里

        # ----------------------------
    # selection & utils (kept mostly same)
    # ----------------------------
    # def select_users(self, round, num_users):
    #     if num_users >= len(self.users):
    #         print("All users are selected")
    #         selected = self.users
    #     else:
    #         selected = list(np.random.choice(self.users, num_users, replace=False))
    #     selected.sort(key=lambda u: u.client_id)
    #     return selected
    def select_users(self, round, ratio):
        if ratio >= 1.0:
            print("All users are selected (ratio >= 1.0)")
            return sorted(self.users, key=lambda u: u.client_id)

        modality_groups = {}
        for user in self.users:
            if isinstance(user.modalities, list):
                m_key = tuple(sorted(user.modalities))
            else:
                m_key = user.modalities

            if m_key not in modality_groups:
                modality_groups[m_key] = []
            modality_groups[m_key].append(user)

        selected_users = []

        for m_key, users_in_group in modality_groups.items():
            n_group = len(users_in_group)

            # 直接用该组总数 * 比例
            n_select = int(n_group * ratio)

            if n_select == 0 and ratio > 0 and n_group > 0:
                n_select = 1

            if n_select > 0:
                selected_indices = np.random.choice(len(users_in_group), n_select, replace=False)
                selected_subset = [users_in_group[i] for i in selected_indices]
                selected_users.extend(selected_subset)

        selected_users.sort(key=lambda u: u.client_id)

        return selected_users

    def train_ae_public(self, seq_len=100):
        """利用公共数据训练自编码器"""
        self.ae_model_server.train()
        modalities_seq, _ = make_seq_batch2(self.public_data, self.batch_size, seq_len=seq_len)
        seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]

        for epoch in range(self.local_epochs):
            idx_end = 0
            while idx_end < seq_len_batch:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len_batch)

                self.optimizer_ae.zero_grad()
                rec_loss = 0.0
                h_latents = {}

               # 只做单模态自重构
                for m in self.modalities_server:
                    x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                    rec = self.ae_model_server(x_m, m)   # 调 forward，内部 encode + decode
                    rec_loss += self.rec_loss_fn(rec, x_m)

                rec_loss.backward()
                self.optimizer_ae.step()

    def train_classifier(self):
        """利用有标签数据训练分类器"""
        self.freeze(self.ae_model_server)
        modalities_seq, y_seq = make_seq_batch2(self.server_train_data, self.batch_size, seq_len=100)
        seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]

        idx_end = 0
        while idx_end < seq_len_batch:
            win_len = np.random.randint(16, 32)
            idx_start = idx_end
            idx_end = min(idx_end + win_len, seq_len_batch)

            self.optimizer_cf.zero_grad()
            latents = []

            # 编码每个模态
            for m in self.modalities_server:
                x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
                _, hidden_seq = self.ae_model_server.encode(x_m, m)
                latents.append(hidden_seq)

            latent_cat = torch.cat(latents, dim=-1)
            # y_batch = torch.from_numpy(y_seq[:, idx_start]).long().to(self.device)
            y_true = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).flatten().long()
            logits = self.cf_model_server(latent_cat)
            print(logits.shape, y_true.shape)
            cls_loss = self.cls_loss_fn(logits, y_true)
            cls_loss.backward()
            self.optimizer_cf.step()
        self.unfreeze(self.ae_model_server)

    # def train_classifier(self):
    #     """利用有标签数据训练分类器"""
    #     self.freeze(self.ae_model_server)
    #     modalities_seq, y_seq = make_seq_batch2(self.server_train_data, self.batch_size, seq_len=100)
    #     seq_len_batch = modalities_seq[self.modalities_server[0]].shape[1]

    #     idx_end = 0
    #     while idx_end < seq_len_batch:
    #         win_len = np.random.randint(16, 32)
    #         idx_start = idx_end
    #         idx_end = min(idx_end + win_len, seq_len_batch)

    #         self.optimizer_cf.zero_grad()

    #         # 编码每个模态
    #         z_m_all = {}
    #         for m in self.modalities_server:
    #             x_m = torch.from_numpy(modalities_seq[m][:, idx_start:idx_end, :]).float().to(self.device)
    #             _, hidden_seq = self.ae_model_server.encode(x_m, m)
    #             z_m_all[m] = hidden_seq    # 直接保持与训练阶段一致

    #         z_fuse = self.fusionNet(z_m_all)
    #         logits = self.cf_model_server(z_fuse)
    #         y_true = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).flatten().long()
    #         cls_loss = self.cls_loss_fn(logits, y_true)
    #         cls_loss.backward()
    #         self.optimizer_cf.step()
    #     self.unfreeze(self.ae_model_server)

    def get_server_state_dicts(self):
        """返回服务器端 AE 和 CF 模型参数"""
        ae_state_dict = self.ae_model_server.state_dict()
        cf_state_dict = self.cf_model_server.state_dict()
        return ae_state_dict, cf_state_dict

    def get_server_ae_state_dicts(self):
        return self.ae_model_server.state_dict()

    def send_ae_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        full_ae_state_dict = self.ae_model_server.state_dict()

        for user in self.users:
            client_modalities = user.modalities if hasattr(user, 'modalities') else []
            if isinstance(client_modalities, str):
                client_modalities = [client_modalities]

            filtered_state_dict = {}
            for key, value in full_ae_state_dict.items():

                for modality in client_modalities:
                    if f"encoders.{modality}." in key or f"decoders.{modality}." in key:
                        filtered_state_dict[key] = value
                        break
            user.set_ae_parameters(filtered_state_dict)

    def aggregate_parameters(self):
        # 同模态聚合
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        server_state_dict = self.ae_model_server.state_dict()

        aggregated_state_dict = {}
        for key in server_state_dict.keys():
            aggregated_state_dict[key] = torch.zeros_like(server_state_dict[key])

        total_samples = sum([user.train_samples for user in self.selected_users])

        for modality in self.modalities_server:
            modality_users = [user for user in self.selected_users
                              if modality in (user.modalities if hasattr(user, 'modalities') else [])]

            if not modality_users:
                continue
            modality_total_samples = sum([user.train_samples for user in modality_users])

            for user in modality_users:
                weight = user.train_samples / modality_total_samples
                user_state_dict = user.get_ae_parameters()

                for key in server_state_dict.keys():
                    if f"encoders.{modality}." in key or f"decoders.{modality}." in key:
                        if key in user_state_dict:
                            aggregated_state_dict[key] += user_state_dict[key] * weight

        new_state_dict = server_state_dict.copy()
        for key in aggregated_state_dict:
            if torch.sum(aggregated_state_dict[key]) != 0:
                new_state_dict[key] = aggregated_state_dict[key]

        self.ae_model_server.load_state_dict(new_state_dict)

    # # ----------------------------
    # # Aggregate parameters (FedAvg-style) from selected_users
    # # ----------------------------
    # def aggregate_parameters(self):
    #     """
    #     Aggregate AE and CF parameters from selected users (weighted by user.train_samples).
    #     Expects each user.get_parameters() -> (ae_params_dict, cf_params_dict)
    #     """
    #     assert (self.selected_users is not None and len(self.selected_users) > 0)

    #     # initialize empty aggregated state dicts (same keys as server)
    #     # Use deepcopy of server to get shapes and keys
    #     server_ae_dict, server_cf_dict = self.get_server_state_dicts()

    #     # zero out
    #     for k in server_cf_dict.keys():
    #         server_cf_dict[k] = torch.zeros_like(server_cf_dict[k])
    #     for m in server_ae_dict.keys():
    #         for k in server_ae_dict[m].keys():
    #             server_ae_dict[m][k] = torch.zeros_like(server_ae_dict[m][k])

    #     # total samples among selected users
    #     total_train = sum([u.train_samples for u in self.selected_users])

    #     # accumulate weighted parameters
    #     for user in self.selected_users:
    #         user_ae_dicts, user_cf_dict = user.get_parameters()  # assume this format
    #         weight = user.train_samples / total_train if total_train > 0 else 1.0 / len(self.selected_users)

    #         # aggregate cf
    #         for k in server_cf_dict.keys():
    #             server_cf_dict[k] += user_cf_dict[k].data.clone() * weight

    #         # aggregate each ae
    #         for m in server_ae_dict.keys():
    #             # user_ae_dicts may be missing some keys if user doesn't have that modality; assume present
    #             for k in server_ae_dict[m].keys():
    #                 server_ae_dicts_val = user_ae_dicts[m][k].data.clone() if (
    #                     m in user_ae_dicts and k in user_ae_dicts[m]) else torch.zeros_like(
    #                     server_ae_dict[m][k])
    #                 server_ae_dict[m][k] += server_ae_dicts_val * weight

    #     # load aggregated params back to server models
    #     self.cf_model_server.load_state_dict(server_cf_dict)
    #     for m in self.ae_models.keys():
    #         self.ae_models[m].load_state_dict(server_ae_dict[m])

    # ----------------------------
    # Add parameters from a single user into server model (weighted add)
    # ----------------------------

    def add_parameters(self, user, ratio):
        """
        Add one user's parameters into server model scaled by ratio.
        Assumes user.get_parameters() returns (ae_params_dict, cf_params_dict)
        """
        user_ae_dicts, user_cf_dict = user.get_parameters()
        # add cf
        server_cf_state = self.cf_model_server.state_dict()
        for k in server_cf_state.keys():
            server_cf_state[k] = server_cf_state[k] + user_cf_dict[k].data.clone() * ratio
        self.cf_model_server.load_state_dict(server_cf_state)

        # add ae per modality
        server_ae_states = {m: self.ae_models[m].state_dict() for m in self.ae_models.keys()}
        for m in server_ae_states.keys():
            for k in server_ae_states[m].keys():
                if m in user_ae_dicts and k in user_ae_dicts[m]:
                    server_ae_states[m][k] = server_ae_states[m][k] + user_ae_dicts[m][k].data.clone() * ratio
                else:
                    # user doesn't have that key (possible if client lacks modality) -> skip or add zeros
                    server_ae_states[m][k] = server_ae_states[m][k] + torch.zeros_like(server_ae_states[m][k])
            self.ae_models[m].load_state_dict(server_ae_states[m])

    # ----------------------------
    # Personalized aggregation helpers (kept similar to your original)
    # ----------------------------
    def persionalized_update_parameters(self, user, ratio):
        """
        Aggregate personalized weights. Assumes user.local_weight_updated is a list of param tensors
        in the same order as server model.parameters() flattened. This function may need customizing
        depending on how you store local_weight_updated for the multi-model case.
        """
        # If you have per-user local_weight_updated stored as (ae_dict, cf_dict), adapt here.
        # For now we keep a conservative implementation: attempt to add user's local_weight_updated in order.
        # WARNING: This depends on user.local_weight_updated structure. If you change User to return dicts,
        # update this function accordingly.
        for server_param, user_param in zip(self.cf_model_server.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        previous_param = copy.deepcopy(list(self.cf_model_server.parameters()))
        # zero out
        for param in self.cf_model_server.parameters():
            param.data = torch.zeros_like(param.data)

        total_train = sum([u.train_samples for u in self.selected_users])
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

        # blend with previous using beta
        for pre_param, param in zip(previous_param, self.cf_model_server.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    # ----------------------------
    # Grad aggregation (if used)
    # ----------------------------
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        # zero grads for server cf and aes (placeholders)
        for param in self.cf_model_server.parameters():
            param.grad = torch.zeros_like(param.data)
        for m in self.ae_models.keys():
            for param in self.ae_models[m].parameters():
                param.grad = torch.zeros_like(param.data)
        # add user grads (user.get_grads should return flattened list matching server param order)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        # WARNING: this requires the user's grads to be in the same flattened order as server params
        # If not, you should adapt by using dict-based grads or change User.get_grads accordingly.
        idx = 0
        for param in self.cf_model_server.parameters():
            param.grad = param.grad + user_grad[idx].clone() * ratio
            idx += 1
        for m in self.ae_models.keys():
            for param in self.ae_models[m].parameters():
                param.grad = param.grad + user_grad[idx].clone() * ratio
                idx += 1

    # ----------------------------
    # Save / Load
    # ----------------------------
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # save both AE dicts and CF
        ae_state_dicts, cf_state_dict = self.get_server_state_dicts()
        torch.save({'ae': ae_state_dicts, 'cf': cf_state_dict}, os.path.join(model_path, "server.pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server.pt")
        assert (os.path.exists(model_path))
        data = torch.load(model_path, map_location='cpu')
        ae_dicts = data['ae']
        cf_dict = data['cf']
        for m in ae_dicts:
            if m in self.ae_models:
                self.ae_models[m].load_state_dict(ae_dicts[m])
        self.cf_model_server.load_state_dict(cf_dict)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server.pt"))

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter, self.num_users)

            for user in self.selected_users:
                user.train_all()

        pass

    def test_clients(self, save=False):
        client_accs = []
        modality_accs = {}   # {mod_name: [acc1, acc2, ...]}

        for c in self.users:
            acc, _ = c.test()
            print(f"→ client {c.client_id} done, acc={acc:.4f}", flush=True)
            client_accs.append(acc)

            # 收集该客户端对应模态的精度
            for mod in c.modalities:  # 例如 ["vision", "text"]
                if mod not in modality_accs:
                    modality_accs[mod] = []
                modality_accs[mod].append(acc)
            print(f"Client {c.client_id} accuracy: {acc:.4f}")

        avg_client_acc = sum(client_accs) / len(client_accs) if len(client_accs) > 0 else 0.0
        avg_modality_acc = {
            mod: sum(acc_list) / len(acc_list)
            for mod, acc_list in modality_accs.items()
        }
        results = {
            "client_accs": client_accs,
            "avg_client_acc": avg_client_acc,
            "avg_modality_acc": avg_modality_acc
        }

        if save:
            if self.pfl:
                os.makedirs(f"results/pfl/{self.dataset}", exist_ok=True)
                with open(f"results/pfl/{self.dataset}/{self.algorithm}_clients_accs.json", "w") as f:
                    import json
                    json.dump(results, f, indent=2)
            else:
                with open(f"results/{self.dataset}/{self.algorithm}_clients_accs.json", "w") as f:
                    import json
                    json.dump(results, f, indent=2)
        return client_accs, avg_client_acc, avg_modality_acc

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

                reps = []
                for m in self.modalities_server:
                    _, hidden_seq = self.ae_model_server.encode(batch_x[m], m)
                    reps.append(hidden_seq.squeeze(0))

                reps_cat = torch.cat(reps, dim=-1)
                outputs = self.cf_model_server(reps_cat)

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

    def freeze(self, sub_model):
        """Freeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = False

    def unfreeze(self, sub_model):
        """Unfreeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = True

    def reset(self):
        device = self.device
        # 1. 重新初始化模型权重
        self.ae_model_server.apply(self._init_weights)
        self.cf_model_server.apply(self._init_weights)
        # 2. 重新初始化优化器
        self.optimizer_ae = torch.optim.Adam(self.ae_model_server.parameters(), lr=self.learning_rate)
        self.optimizer_cf = torch.optim.Adam(self.cf_model_server.parameters(), lr=self.learning_rate)

        for user in self.users:
            user.ae_model.apply(self._init_weights)
            user.cf_model.apply(self._init_weights)
            user.optimizer_ae = torch.optim.Adam(user.ae_model.parameters(), lr=self.learning_rate)
            user.optimizer_cf = torch.optim.Adam(user.cf_model.parameters(), lr=self.learning_rate)
        # 3. 清空训练记录
        self.rs_train_acc = []
        self.rs_train_loss = []
        self.rs_glob_acc = []
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_glob_acc_per = []

        # 4. 清空用户选择
        self.selected_users = []

        # 5. 可选：清空全局训练样本统计
        self.total_train_samples = 0

        print("Server has been reset.")

    # 辅助函数：初始化权重

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)

    def save_results(self):
        if not self.pfl:
            save_dir = os.path.join("results", self.dataset)
        else:
            save_dir = os.path.join("results/pfl", self.dataset)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{self.algorithm}.svg")

        # 新增：保存精度数据到 JSON 文件
        json_save_path = os.path.join(save_dir, f"{self.algorithm}_acc.json")
        acc_data = {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "global_accuracy": self.rs_glob_acc,
            "global_f1": self.rs_glob_f1,
            "train_accuracy": self.rs_train_acc if hasattr(self, 'rs_train_acc') else [],
            "rounds": list(range(1, len(self.rs_glob_acc) + 1))
        }
        with open(json_save_path, 'w') as f:
            json.dump(acc_data, f, indent=2)

        print(f"Accuracy data saved to {json_save_path}")

        rounds = list(range(1, len(self.rs_glob_acc) + 1))

        plt.figure(figsize=(10, 4))

        # 左图：Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(rounds, self.rs_glob_acc, label="Global Acc")
        if self.rs_train_acc:
            plt.plot(rounds, self.rs_train_acc, label="Train Acc")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)

        # 右图：Loss
        plt.subplot(1, 2, 2)
        plt.plot(rounds, self.rs_train_loss, label="Train Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, format="svg")
        plt.close()

        print(f"Results saved to {save_path}")

    def check_model_mode(self, model, indent=0):

        # 打印当前模块的信息
        indent_str = "  " * indent
        module_name = model.__class__.__name__  # 模块类名
        print(f"{indent_str}- {module_name}: training={model.training}")

        # 递归检查子模块
        for child in model.children():
            self.check_model_mode(child, indent + 1)
