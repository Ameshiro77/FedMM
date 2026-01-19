# ==========================
#     ServerProto.py
# ==========================

import torch
import numpy as np
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.userproto import UserProto
from utils.model_utils import *
from FLAlgorithms.trainmodel.ae_model import *

class FedProto(Server):

    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1, pfl=False):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times, pfl)

        # 实例化客户端
        for i in range(self.total_users):
            
            client_modals = modalities[i]
            if isinstance(client_modals, list):  # 支持多模态客户端
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:  # 单模态
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size
            )
            client_cf = MLP(rep_size , n_classes)

            if pfl:
                user = UserProto(
                i, self.clients_train_data_list[i], self.clients_test_data_list[i], self.public_data,
                client_ae, client_cf, client_modals,
                batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio
            )
            else:
                user = UserProto(
                    i, self.clients_train_data_list[i], self.test_data, self.public_data,
                    client_ae, client_cf, client_modals,
                    batch_size, learning_rate, beta, lamda,
                    local_epochs, label_ratio
                )
            self.users.append(user)

        # 全局 prototype
        self.global_prototypes = {}

    # ----------------------------------------------------
    # 聚合客户端原型
    # ----------------------------------------------------
    def aggregate_prototypes(self, proto_dicts):
        """
        输入：[(proto_dict, weight_dict), ...]
        输出：全局原型
        """
        agg = {}        # sum(n_i * p_i)
        count = {}      # sum(n_i)

        for proto, weights in proto_dicts:
            for cls, p in proto.items():
                w = weights[cls]
                if cls not in agg:
                    agg[cls] = p * w
                    count[cls] = w
                else:
                    agg[cls] += p * w
                    count[cls] += w

        # 最终加权平均
        global_proto = {}
        for cls in agg:
            global_proto[cls] = (agg[cls] / count[cls]).detach()

        return global_proto

    # ----------------------------------------------------
    # 主循环
    # ----------------------------------------------------
    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print(f"\n========== Global Round {glob_iter+1}/{self.num_glob_iters} ==========")

     
            proto_dicts = []
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                proto, weights = user.upload_prototype()
                proto_dicts.append((proto, weights))

            # ------- Step 2: 聚合 prototype -------
            self.global_prototypes = self.aggregate_prototypes(proto_dicts)

            # ------- Step 3: 客户端使用 global prototype 训练 classifier -------
            for user in self.selected_users:
                user.train_cf_proto(self.global_prototypes)

            self.train_classifier()  # freeze

            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)
        
        accs = self.test_clients()
        print("Test accuracy: ", accs)
        print("all saved")
        self.save_results()
