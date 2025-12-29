#!/bin/bash

COMMON_ARGS="--dataset mhealth --global_rounds 200"

# 定义你的算法列表
ALGOS=("fedprox" "fedavg")
# ALGOS=("fedprox" "fedmekt" "fedproto" "fedavg" "fedcent" "feddtg" "fedprop")

echo "Starting experiments with GNU Parallel..."

# === 关键命令解释 ===
# -j 2          : 最多同时运行 2 个任务 (根据你的显卡/CPU能力调整)
# --eta         : 显示预计完成时间
# --joblog log  : 记录任务运行日志
# {}            : 会被替换成 ALGOS 里的每一个算法名
# :::           : 后面跟参数列表

parallel -j 2 --eta --line-buffer \
    "python main.py $COMMON_ARGS --algo {}" ::: "${ALGOS[@]}"

echo "All algorithms finished."