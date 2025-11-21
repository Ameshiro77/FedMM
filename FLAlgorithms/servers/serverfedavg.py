from FLAlgorithms.users.userbase import User
# from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import *
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.users.userfedavg import UserFedAvg
import json
import codecs
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
from torch import nn, optim
from sklearn.metrics import f1_score
import sys


class FedAvg(Server):
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1):
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times)

        self.total_users = len(modalities)

        for i in range(self.total_users):

            client_modals = modalities[i]
            if isinstance(client_modals, list):  # 支持多模态客户端
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:  # 单模态
                input_sizes_client = {client_modals: input_sizes[client_modals]}

            # 实例化该客户端的 AE
            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size
            )
            client_cf = MLP(rep_size, n_classes)
            user = UserFedAvg(
                i, self.clients_train_data_list[i], self.test_data, self.public_data, client_ae, client_cf,
                client_modals, batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio=label_ratio)
            print("client", i, "modals:", client_modals)
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")

            self.send_ae_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)

            for user in self.selected_users:
                user.train_ae()

            self.aggregate_parameters()
            self.send_ae_parameters()

            

            for user in self.users:
                user.train_cf()

            # self.train()
            self.train_classifier()  # freeze

            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)
            
            # =========================================================================================
            # === 总通信量统计 ===
            # total_uplink_bytes = 0
            # total_downlink_bytes = 0
            
            # 计算总通信量（假设每轮通信量相同）
            # server_ae_params = self.ae_model_server.state_dict()
            # one_round_downlink = sum(param.numel() * 4 for param in server_ae_params.values())
            # one_round_uplink = 0
            
            # for user in self.users:
            #     user_ae_params = user.get_ae_parameters()
            #     one_round_uplink += sum(param.numel() * 4 for param in user_ae_params.values())
            
            # total_uplink_bytes = one_round_uplink * self.num_glob_iters
            # total_downlink_bytes = one_round_downlink * self.num_glob_iters 
            
            # print(f"\n=== FedAvg 通信量统计 ===")
            # print(f"总轮数: {self.num_glob_iters}")
            # print(f"总上行通信量: {total_uplink_bytes/1024/1024:.2f} MB")
            # print(f"总下行通信量: {total_downlink_bytes/1024/1024:.2f} MB")
            # print(f"总通信量: {(total_uplink_bytes+total_downlink_bytes)/1024/1024:.2f} MB")
            # print(f"平均每轮上行: {total_uplink_bytes/self.num_glob_iters/1024:.2f} KB")
            # print(f"平均每轮下行: {total_downlink_bytes/self.num_glob_iters/1024:.2f} KB")
            # print(f"平均每轮通信量: {(total_uplink_bytes+total_downlink_bytes)/self.num_glob_iters/1024:.2f} KB")
            # exit()
        
        accs = self.test_clients()
        print("Test accuracy: ", accs)
        self.save_results()
       
