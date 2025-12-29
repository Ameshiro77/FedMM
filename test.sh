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
#     --algo fedprop
    
python main.py \
    --dataset opp \
    --global_rounds 200 \
    --local_epochs 2 \
    --numusers 0.5 \
    --optimizer Adam \
    --algo fedprop \
    --pfl