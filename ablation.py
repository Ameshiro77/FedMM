#!/usr/bin/env python
import argparse
import copy

# 直接引用你的 main.py
from main import main

import os
import json


def save_ablation_json(
    dataset,
    scheme,
    rs_glob_acc,
    avg_client_acc,
    avg_modality_acc,
    pfl=False,
    exp="ab"
):
    
    dir_dict = {
        "ab": "ablation",
        "dist": "hyper_dist",
        "align": "hyper_align"
    }
    out_dir = dir_dict[exp]
    
    if pfl:
        save_dir = f"./results/pfl/{out_dir}"
    else:
        save_dir = f"./results/{out_dir}"

    os.makedirs(save_dir, exist_ok=True)

    scheme_name = scheme.replace(" ", "_")
    save_path = os.path.join(
        save_dir, f"{dataset}_{scheme_name}_ab.json"
    )

    record = {
        "scheme": scheme,
        "dataset": dataset,
        "server_acc_curve": rs_glob_acc,
        "client_avg_acc": avg_client_acc,
        "modality_acc": avg_modality_acc
    }

    with open(save_path, "w") as f:
        json.dump(record, f, indent=4)

    print(f"[Saved] Ablation JSON => {save_path}")


def get_ablation_settings(exp="ab"):

    if exp == "ab":
        return {
            "fedprop":{
                "server_dist_weight": 0.5,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
            # "wo_server_dist": {
            #     "server_dist_weight": 0.0,
            #     "server_align_weight": 0.05,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 1.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": 0.5,
            # },
            
            # "wo_server_align": {
            #     "server_dist_weight": 0.8,
            #     "server_align_weight": 0.0,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 1.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": 0.5,
            # },

            # "wo_client_align": {
            #     "server_dist_weight": 0.5,
            #     "server_align_weight": 0.0,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 0.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": 0.5,
            # },

            # "wo_client_dist": {
            #     "server_dist_weight": 0.5,
            #     "server_align_weight": 0.0,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 1.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": 0.0,
            # },
            
            # "wo_client_dyn": {
            #     "server_dist_weight": 0.5,
            #     "server_align_weight": 0.0,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 1.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": -1.0,
            # },


            # "wo_client_ab": {
            #     "algo": "fedab",
            #     "server_dist_weight": 1.0,
            #     "server_align_weight": 0.05,
            #     "client_orth_weight": 0.1,
            #     "client_align_weight": 1.0,
            #     "client_reg_weight": 0.01,
            #     "client_logits_weight": 0.5,
            # }
        }

    if exp == "dist":

        return {

            "fedprop_serverdist_0.0": {
                "server_dist_weight": 0.0,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.1": {
                "server_dist_weight": 0.1,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.2": {
                "server_dist_weight": 0.2,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.3": {
                "server_dist_weight": 0.3,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.4": {
                "server_dist_weight": 0.4,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.5": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.6": {
                "server_dist_weight": 0.6,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.7": {
                "server_dist_weight": 0.7,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.8": {
                "server_dist_weight": 0.8,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_0.9": {
                "server_dist_weight": 0.9,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serverdist_1.0": {
                "server_dist_weight": 1.0,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
        }

    if exp == "align":
        return {
            "fedprop": {
                "server_dist_weight": 0.8,
                "server_align_weight": 0.0,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },

            "fedprop_serveralign_0.00": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.00,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
            "fedprop_serveralign_0.05": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.05,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
            "fedprop_serveralign_0.1": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.1,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
            "fedprop_serveralign_0.15": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.15,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
            "fedprop_serveralign_0.2": {
                "server_dist_weight": 0.5,
                "server_align_weight": 0.2,
                "client_orth_weight": 0.1,
                "client_align_weight": 1.0,
                "client_reg_weight": 0.01,
                "client_logits_weight": 0.5,
            },
        }

        # ============


if __name__ == "__main__":
    import copy

    import torch
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(42)
    EVAL_WIN = 100
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="opp")
    parser.add_argument("--algo", type=str, default="FedProp")
    parser.add_argument("--model", type=str, default="dnn")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--global_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--numusers", type=float, default=0.5)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--exp", type=str, default="ab")

    args = parser.parse_args()

    # === 标记为消融模式 ===
    args.ablation = True
    args.pfl = True

    client_modalities_dict = {
        "mhealth": [5, 5, 5],
        "opp": [5, 5],
        "ur_fall": [5, 5, 5],
        "uci_har": [5, 5],
        "hapt": [5, 5],
        "pamap2": [5, 5, 5]
    }

    ablations = get_ablation_settings(args.exp)

    if args.dataset == "ur_fall":
        args.global_rounds = 200

    for scheme_name, weights in ablations.items():
        print("=" * 80)
        print(f"Running ablation: {scheme_name}")
        print(weights)

        ab_args = copy.deepcopy(args)

        for k, v in weights.items():
            setattr(ab_args, k, v)

        rs_glob_acc, avg_client_acc, avg_modality_acc = main(
            dataset=ab_args.dataset,
            algorithm=ab_args.algo,
            model=ab_args.model,
            batch_size=ab_args.batch_size,
            learning_rate=ab_args.learning_rate,
            num_glob_iters=ab_args.global_rounds,
            local_epochs=ab_args.local_epochs,
            optimizer=ab_args.optimizer,
            num_users=ab_args.numusers,
            client_modalities_dict=client_modalities_dict,
            times=1,
            pfl=ab_args.pfl,
            args=ab_args
        )

        save_ablation_json(
            dataset=ab_args.dataset,
            scheme=scheme_name,
            rs_glob_acc=rs_glob_acc,
            avg_client_acc=avg_client_acc,
            avg_modality_acc=avg_modality_acc,
            pfl=ab_args.pfl,
            exp=args.exp
        )
