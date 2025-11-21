import copy
import torch
import torch.nn as nn
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.usermekt import UserMEKT
from utils.model_utils import make_seq_batch, get_seg_len, load_data, client_idxs
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
import numpy as np

class FedMEKT(Server):
    
    def __init__(self, dataset, algorithm, input_sizes, rep_size, n_classes,
                 modalities, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, label_ratio=0.1):
        
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times)
        
        
        # 初始化客户端
        self.total_users = len(modalities)
        self.users = []
        
        for i in range(self.total_users):
            client_modals = modalities[i]
            if isinstance(client_modals, list):
                input_sizes_client = {m: input_sizes[m] for m in client_modals}
            else:
                input_sizes_client = {client_modals: input_sizes[client_modals]}
            
            # 客户端AE模型
            client_ae = SplitLSTMAutoEncoder(
                input_sizes=input_sizes_client,
                representation_size=rep_size
            )
            client_cf = MLP(rep_size, n_classes)
            
            user = UserMEKT(
                i, self.clients_train_data_list[i], self.test_data, self.public_data, 
                client_ae, client_cf, client_modals, batch_size, learning_rate,
                beta, lamda, local_epochs
            )
            self.users.append(user)
        
    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")

            self.selected_users = self.select_users(glob_iter, self.num_users)

            for user in self.selected_users:
                user.train_ae_distill(self.local_epochs, self.ae_model_server)

            for user in self.users:
                user.train_cf()

            # self.train()
            self.train_classifier()  # freeze

            loss, acc, f1 = self.test_server()
            self.rs_train_loss.append(loss)
            self.rs_glob_acc.append(acc)
            self.rs_glob_f1.append(f1)
        accs = self.test_clients()
        print("Test accuracy: ", accs)
        self.save_results()

    