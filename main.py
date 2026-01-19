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
         local_epochs, optimizer, num_users, client_modalities_dict, times, pfl, args): \

    torch.manual_seed(0)
    np.random.seed(42)
    EVAL_WIN = 100
    rs_glob_acc, avg_client_acc, avg_modality_acc = [], 0, {}
    label_ratio = args.label_ratio
    
    if dataset == 'opp':
        rep_size = 10
        batch_size = 256
    elif dataset == 'mhealth':
        rep_size = 10
        batch_size = 256
    elif dataset == 'ur_fall':
        rep_size = 10
        batch_size = 128
    elif dataset == "hapt":
        rep_size = 32
        batch_size = 256
    elif dataset == "pamap2":
        rep_size = 10
        batch_size = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = load_data(dataset)

    print("============================================================================================")
    print("train data:", len(data[0]['y']), "test data:", len(data[1]['y']))
    print("public data:", len(data[2]['y']))
    print("============================================================================================")

    server_test = data[1]

    train_server, _ = split_server_train(data[0], dataset=dataset)
    
    # server_mb = get_data_size_mb(train_server)

    # print(f"[Server Data Size] {server_mb:.2f} MB")
    # exit()
        
    # if set(train_server["y"]) == set(server_test["y"]):
    #     break

    modalities = [m for m in train_server.keys() if m != "y"]
    num_clients_per_modality = client_modalities_dict[dataset]
    client_modalities_list = []
    for m, n in zip(modalities, num_clients_per_modality):
        client_modalities_list += [m] * n

    input_sizes = {m: train_server[m].shape[1] for m in train_server if m != "y"}
    print(input_sizes)
    # n_classes = len(set(train_server["y"]))
    all_labels = np.concatenate([data[0]["y"], data[1]["y"]], axis=0)
    n_classes = len(np.unique(all_labels))

    for i in range(times):

        print("---------------Running time:------------", i)

        # select algo

        if algorithm.lower() == "fedprop":
            server = FedProp(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,
                             num_users=num_users, times=times, pfl=pfl, label_ratio=label_ratio,args=args)

        if algorithm.lower() == "fedavg":
            server = FedAvg(dataset, algorithm, input_sizes=input_sizes,
                            modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                            batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                            num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                            num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedprox":
            server = FedProx(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                             num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedcent":
            server = FedCent(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                             num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedcream":
            server = FedCream(dataset, algorithm, input_sizes=input_sizes,
                              modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                              batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                              num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                              num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedproto":
            server = FedProto(dataset, algorithm, input_sizes=input_sizes,
                              modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                              batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                              num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                              num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedmekt":
            server = FedMEKT(dataset, algorithm, input_sizes=input_sizes,
                             modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                             batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                             num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                             num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "feddf":
            server = FedDF(dataset, algorithm, input_sizes=input_sizes,
                           modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                           batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                           num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                           num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "fedab":
            server = FedNoDis(dataset, algorithm, input_sizes=input_sizes,
                              modalities=client_modalities_list, rep_size=rep_size*2, n_classes=n_classes,
                              batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                              num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                              num_users=num_users, times=times, pfl=pfl,args=args)

        if algorithm.lower() == "fedpropgen":
            server = FedPropGen(dataset, algorithm, input_sizes=input_sizes,
                                modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                                batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                                num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                                num_users=num_users, times=times, pfl=pfl)

        if algorithm.lower() == "feddtg":
            server = FedDTG(dataset, algorithm, input_sizes=input_sizes,
                            modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                            batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                            num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                            num_users=num_users, times=times, pfl=pfl)
            
        if algorithm.lower() == "fedgao":
            server = FedGao(dataset, algorithm, input_sizes=input_sizes,
                            modalities=client_modalities_list, rep_size=rep_size, n_classes=n_classes,
                            batch_size=batch_size, learning_rate=learning_rate, beta=1.0, lamda=0.0,
                            num_glob_iters=num_glob_iters, local_epochs=local_epochs, optimizer=optimizer,label_ratio=label_ratio,
                            num_users=num_users, times=times, pfl=pfl, args=args)

        server.reset()
        
        
        if (algorithm.lower() == "fedprop" or algorithm.lower() == "fedab") and (args.ablation or args.manual):
            rs_glob_acc, avg_client_acc, avg_modality_acc = server.train()
            
            save_dir = ""
            save_name = ""

            # --- 分支 1: 消融实验模式 (优先) ---
            if args.ablation_name:
                scheme_name = args.ablation_name
                dir_dict = {
                    "ab": "ablation",
                    "dist": "hyper_dist",
                    "align": "hyper_align"
                }
                out_dir = dir_dict.get(args.exp_type, "ablation")
                
                if pfl:
                    save_dir = f"./results/pfl/{out_dir}"
                else:
                    save_dir = f"./results/{out_dir}"
                
                safe_scheme_name = scheme_name.replace(" ", "_")
                save_name = f"{dataset}_{safe_scheme_name}_ab.json"

            # --- 分支 2: 手动模式 (当没有 ablation_name 但有 manual 标志时) ---
            elif args.manual:
                if pfl:
                    save_dir = "./results/pfl/manual"
                else:
                    save_dir = "./results/manual"
                
                # 按你的要求命名: fedprop_{args.manual}_ab.json
                save_name = f"fedprop_{args.manual}_ab.json"

            # --- 执行保存 ---
            if save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)
                
                record = {
                    "scheme": args.ablation_name if args.ablation_name else f"manual_{args.manual}",
                    "dataset": dataset,
                    "server_acc_curve": rs_glob_acc,
                    "client_avg_acc": avg_client_acc,
                    "modality_acc": avg_modality_acc
                }
                
                with open(save_path, "w") as f:
                    json.dump(record, f, indent=4)
                print(f"[Saved] Result JSON => {save_path}")

        else:
            # 普通训练模式 (不存特定JSON)
            server.train()
        
    return rs_glob_acc, avg_client_acc, avg_modality_acc
    #     if algorithm.lower() == "fedprop" and args.ablation:
    #         rs_glob_acc, avg_client_acc, avg_modality_acc = server.train()
    #     else:
    #         server.train()
        
    # return rs_glob_acc, avg_client_acc, avg_modality_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.backends.cudnn.enabled = False
    # opp:361620
    parser.add_argument("--dataset", type=str, default="opp")
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "cnn"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--algo", type=str, default="FedProp")
    parser.add_argument("--numusers", type=float, default=0.5, help="Number of Users per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--global_rounds", type=int, default=100, help="Number of global rounds")
    parser.add_argument("--label_ratio", type=float, default=0.1, help="")
    parser.add_argument("--pfl", default=False, action='store_true')
    
    # === 【新增】消融实验专用参数 (以前你是硬编码的，现在要暴露出来) ===
    parser.add_argument("--server_dist_weight", type=float, default=0.1)
    parser.add_argument("--server_align_weight", type=float, default=0.0)
    parser.add_argument("--client_orth_weight", type=float, default=0.1)
    parser.add_argument("--client_align_weight", type=float, default=0.7)
    parser.add_argument("--client_reg_weight", type=float, default=0.01)
    parser.add_argument("--client_logits_weight", type=float, default=0.5)
    
    # === 【新增】用于标记这次运行是不是消融，以及消融的名字 ===
    parser.add_argument("--ablation_name", type=str, default="", help="If set, saves to ablation folder")
    parser.add_argument("--manual", type=str, default="", help="If set, ab name")
    parser.add_argument("--exp_type", type=str, default="ab", choices=["ab", "dist", "align"])
    
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
        "ur_fall": [5, 5, 5],
    }

    # === 【逻辑修改】根据是否传入 ablation_name 来设置 args.ablation ===
    if args.ablation_name or args.manual:
        args.ablation = True # 开启消融模式逻辑
    else:
        args.ablation = False

    print("=" * 80)
    print(f"Algorithm: {args.algo}")
    print(f"Ablation Name: {args.ablation_name if args.ablation_name else 'None'}")
    print(f"Weights -> Dist:{args.server_dist_weight}, Align:{args.server_align_weight}, Logits:{args.client_logits_weight}")
    print("=" * 80)
    
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
        times=args.times,
        pfl=args.pfl,
        args=args
    )
