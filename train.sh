#!/bin/bash
#acc_delta  dataset
#hybrid  alpha  rew  method algo
# ent_coef norm_xi LSTM
#ALGOS=("fedprop" "fedprox" "fedcent" "fedmekt" "fedproto" "fedavg")

COMMON_ARGS="\
    --dataset hapt \
    --global_rounds 100"

ALGOS=("fedprop" "fedprox" "fedmekt" "fedproto" "fedavg" "fedcent")

for algo in "${ALGOS[@]}"; do
    echo "Running algorithm: $algo"
    python main.py $COMMON_ARGS --algo "$algo"
    echo "Finished: $algo"
    echo "------------------------"
done