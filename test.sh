#!/bin/bash
#acc_delta  dataset
#hybrid  alpha  rew  method algo
# ent_coef norm_xi LSTM
# mhealth opp hapt ur_fall

# python main.py \
#     --dataset mhealth \
#     --global_rounds 100 \
#     --local_epochs 2 \
#     --numusers 0.5 \
#     --algo fedgao \
    # --pfl

# python main.py \
#     --dataset opp \
#     --global_rounds 100 \
#     --local_epochs 2 \
#     --numusers 0.5 \
#     --optimizer Adam \
#     --algo fedgao \
#     --pfl

python main.py \
    --dataset ur_fall \
    --global_rounds 200 \
    --local_epochs 2 \
    --numusers 0.5 \
    --optimizer Adam \
    --algo fedprop \
    --server_dist_weight 0.1 \
    --server_align_weight 0.00 \
    --client_orth_weight 0.1 \
    --client_align_weight 0.7 \
    --client_reg_weight 0.01 \
    --client_logits_weight 0.5 \
    --pfl