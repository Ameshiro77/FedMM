import copy
import torch
import torch.nn as nn
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.usermekt import UserMEKT
from utils.model_utils import make_seq_batch, get_seg_len, load_data, client_idxs
from FLAlgorithms.trainmodel.ae_model import SplitLSTMAutoEncoder, MLP
import numpy as np

class FedMEKT(Server):
    def __init__(self, train_data, test_data, public_data, device, dataset, algorithm, 
                 input_sizes, rep_size, n_classes, modalities, batch_size, learning_rate, 
                 beta, lamda, num_glob_iters, local_epochs, optimizer, num_users, times):
        
        super().__init__(dataset, algorithm, input_sizes, rep_size, n_classes,
                         modalities, batch_size, learning_rate, beta, lamda,
                         num_glob_iters, local_epochs, optimizer, num_users, times)
        
        self.device = device
        self.public_data = public_data
        self.test_data = test_data
        
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
                i, self.clients_train_data_list[i], test_data, public_data, 
                client_ae, client_cf, client_modals, batch_size, learning_rate,
                beta, lamda, local_epochs
            )
            self.users.append(user)

    