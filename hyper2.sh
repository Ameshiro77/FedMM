#!/bin/bash

# 定义两个并行数组：一个对应epoch值，一个对应manual名称
epochs=(1 2 3 4 5)
manuals=("ep1" "ep2" "ep3" "ep4" "ep5")

# 循环执行
for i in {0..4}; do  # 数组索引从0开始
    epoch=${epochs[$i]}
    manual=${manuals[$i]}
    
    echo "========================================"
    echo "开始运行: local_epochs=$epoch, manual=$manual"
    echo "========================================"
    
    python main.py \
        --dataset ur_fall \
        --global_rounds 200 \
        --local_epochs $epoch \
        --numusers 0.5 \
        --optimizer Adam \
        --algo fedprop \
        --pfl \
        --manual "$manual" \
        --server_dist_weight 0.1 \
        --server_align_weight 0.00 \
        --client_orth_weight 0.1 \
        --client_align_weight 0.7 \
        --client_reg_weight 0.00 \
        --client_logits_weight 0.5
    
    sleep 2  # 可选间隔，确保稳定
done

echo "所有实验运行完毕！"