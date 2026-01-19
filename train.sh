#!/bin/bash
#acc_delta  dataset
#hybrid  alpha  rew  method algo
# ent_coef norm_xi LSTM
#ALGOS=("fedprop" "fedprox" "fedcent" "fedmekt" "fedproto" "fedavg")

# COMMON_ARGS="\
#     --dataset ur_fall \
#     --local_epochs 2 \
#     --optimizer Adam \
#     --numusers 0.5 \
#     --global_rounds 200 \
#     --pfl "


# COMMON_ARGS="\
#     --dataset mhealth \
#     --local_epochs 2 \
#     --optimizer Adam \
#     --numusers 0.5 \
#     --pfl \
#     --global_rounds 100 "

COMMON_ARGS="\
    --dataset opp \
    --local_epochs 2 \
    --optimizer Adam \
    --numusers 0.5 \
    --pfl \
    --global_rounds 100 "

# ALGOS=("fedprox" "fedavg" "fedprop")
ALGOS=("fedproto" "fedprop")
# ALGOS=("fedprox" "fedmekt" "fedproto" "fedavg" "fedcent" "feddtg" "fedprop")

for algo in "${ALGOS[@]}"; do
    echo "Running algorithm: $algo"
    python main.py $COMMON_ARGS --algo "$algo"
    echo "Finished: $algo"
    echo "------------------------"
done