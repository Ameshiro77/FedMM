#!/usr/bin/env python
import scipy.io
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.trainmodel.ae_model import *
from utils.model_utils import *

from FLAlgorithms.servers import *

# from utils.plot_utils import *
import torch
torch.manual_seed(0)
np.random.seed(42)
EVAL_WIN = 100

def main(dataset, algorithm, model, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer, num_users, client_modalities_dict, times): \


    if dataset == 'opp':
        rep_size = 10
    elif dataset == 'mhealth':
        rep_size = 4
    elif dataset == 'ur_fall':
        rep_size = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = load_data(dataset)
    
    print("============================================================================================")
    print("train data:", len(data[0]['y']), "test data:", len(data[1]['y']))
    print("public data:", len(data[2]['y']))
    print("============================================================================================")
    
    server_test = data[1]
    while True:
        train_server, _ = split_server_train(data[0], dataset=dataset)
        if set(train_server["y"]) == set(server_test["y"]):
            break

    modalities = [m for m in train_server.keys() if m != "y"]
    num_clients_per_modality = client_modalities_dict[dataset]
    client_modalities_list = []
    for m, n in zip(modalities, num_clients_per_modality):
        client_modalities_list += [m] * n

    input_sizes = {m: train_server[m].shape[1] for m in train_server if m != "y"}
    print(input_sizes)
    n_classes = len(set(train_server["y"]))

    for i in range(times):
        
        print("---------------Running time:------------", i)

        # select algo

        if algorithm.lower() == "fedprop":
            server = FedProp(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
            
        if algorithm.lower() == "fedavg":
            server = FedAvg(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
        
        if algorithm.lower() == "fedprox":
            server = FedProx(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
        
        if algorithm.lower() == "fedcent":
            server = FedCent(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
        
        if algorithm.lower() == "fedmoon":
            server = FedMoon(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
        
        if algorithm.lower() == "fedmekt":
            server = FedMEKT(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times)
        
        server.reset()
        server.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.backends.cudnn.enabled = False
    # opp:361620
    parser.add_argument("--dataset", type=str, default="opp", choices=["mhealth", "opp", "ur_fall"])
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "cnn"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algo", type=str, default="FedProp")
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--global_rounds", type=int, default=100, help="Number of global rounds")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algo))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.global_rounds))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    # client_modalities_dict = {
    #     "mhealth": [10, 10, 10],
    #     "opp": [15, 15],
    #     "ur_fall": [10, 10, 10]
    # }
    client_modalities_dict = {
        "mhealth": [5, 5, 5],
        "opp": [5, 5],
        "ur_fall": [5, 5, 5]
    }
    args.numusers = sum(client_modalities_dict[args.dataset])

    main(
        dataset=args.dataset,
        algorithm=args.algo,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.global_rounds,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        num_users=args.numusers,
        client_modalities_dict=client_modalities_dict,
        times=args.times
    )
