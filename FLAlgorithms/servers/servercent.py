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


class FedCent(Server):
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
            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size
            )

            client_cf = MLP(rep_size, n_classes)
            user = User(
                i, self.clients_train_data_list[i], self.test_data, self.public_data, client_ae, client_cf,
                client_modals, batch_size, learning_rate, beta, lamda,
                local_epochs, label_ratio=label_ratio)
            print("client", i, "modals:", client_modals)
            self.users.append(user)

    def train(self):

        for glob_iter in range(self.num_glob_iters):
            print(f"\n--- Global Round [{glob_iter+1}/{self.num_glob_iters}] ---")
            self.train_classifier()
            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)
            
            for user in self.users:
                user.train_ae()
                user.train_cf()
            
        print("Test F1: ", self.rs_glob_f1)
        print("Test accuracy: ", self.rs_glob_acc)
        accs = self.test_clients()
        print("Test accuracy: ", accs)
        self.save_results()
