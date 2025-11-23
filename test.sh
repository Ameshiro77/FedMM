#!/bin/bash
#acc_delta  dataset
#hybrid  alpha  rew  method algo
# ent_coef norm_xi LSTM
# mhealth opp hapt

python main.py \
    --dataset mhealth  \
    --global_rounds 200 \
    --algo fedprox
