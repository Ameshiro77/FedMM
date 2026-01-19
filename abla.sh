#!/bin/bash

# ================= 定义实验运行函数 =================
# 参数顺序: 
# 1: ablation_name (实验名)
# 2: algo (算法名)
# 3: s_dist (服务端蒸馏权重)
# 4: s_align (服务端对齐权重)
# 5: c_orth (客户端正交权重)
# 6: c_align (客户端对齐权重)
# 7: c_reg (客户端正则权重)
# 8: c_logits (客户端逻辑值权重)

run_exp() {
    name=$1
    algo=$2
    s_dist=$3
    s_align=$4
    c_orth=$5
    c_align=$6
    c_reg=$7
    c_logits=$8

    echo "============================================================"
    echo "正在运行消融实验: $name"
    echo "算法: $algo | S_Dist: $s_dist | C_Align: $c_align | C_Logits: $c_logits"
    echo "============================================================"

    python main.py \
        --dataset ur_fall \
        --global_rounds 200 \
        --local_epochs 2 \
        --numusers 0.5 \
        --optimizer Adam \
        --pfl \
        --exp_type ab \
        --ablation_name "$name" \
        --algo "$algo" \
        --server_dist_weight $s_dist \
        --server_align_weight $s_align \
        --client_orth_weight $c_orth \
        --client_align_weight $c_align \
        --client_reg_weight $c_reg \
        --client_logits_weight $c_logits

    if [ $? -ne 0 ]; then
        echo "❌ 错误：实验 $name 运行失败！"
        # exit 1  # 如果希望出错继续跑下一个，请注释掉这行
    fi
    
    sleep 2 # 冷却时间
}

# ================= 执行消融实验组 =================

# 1. FedProp (基准/Ours)
# 对应 Python 注释中的 "fedprop"
run_exp "fedprop" "fedprop" \
    0.1  0.00 \
    0.1  0.7  0.01  0.5

# 2. w/o Server Dist (去掉服务端蒸馏)
# 对应 Python 注释中的 "wo_server_dist": server_dist_weight=0.0
run_exp "wo_server_dist" "fedprop" \
    0.0  0.00 \
    0.1  0.7  0.01  0.5

# 3. w/o Client Align (去掉客户端对齐)
# 对应 Python 注释中的 "wo_client_align": client_align_weight=0.0
run_exp "wo_client_align" "fedprop" \
    0.1  0.00 \
    0.1  0.0  0.01  0.5

# 4. w/o Client Dist (去掉客户端 KD)
# 对应 Python 注释中的 "wo_client_dist": client_logits_weight=0.0
run_exp "wo_client_dist" "fedprop" \
    0.1  0.00 \
    0.1  0.7  0.01  0.0

# 5. FedAB (对比算法)
# 对应 Python 代码中的 "wo_client_ab": algo="fedab"
run_exp "wo_client_ab" "fedab" \
    0.1  0.00 \
    0.1  0.7  0.01  0.5

echo "✅ 所有消融实验运行完毕！"