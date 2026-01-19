#!/bin/bash

# 定义参数值的数组
# dist_weights=(0.0)
# align_weights=(0.0)
dist_weights=(0.1 0.3 0.5 0.7 0.9)
align_weights=(0.1 0.3 0.5 0.7 0.9)
# dist_weights=(0.0 0.2 0.4 0.6 0.8)
# align_weights=(0.0 0.2 0.4 0.6 0.8)

# 外层循环：遍历 server_dist_weight
for dist in "${dist_weights[@]}"; do
    
    # 内层循环：遍历 client_align_weight
    for align in "${align_weights[@]}"; do
        
        # 1. 构造唯一的 manual 名称
        # 格式示例: sdist_0.1_calign_0.3
        # 这样生成的 json 会是: fedprop_sdist_0.1_calign_0.3_ab.json，非常清晰
        manual_name="sdist_${dist}_calign_${align}"
        
        echo "========================================"
        echo "正在运行组合: Server Dist = $dist, Client Align = $align"
        echo "Manual Name : $manual_name"
        echo "========================================"
        
        # 2. 运行 Python 命令
        # 注意：这里把循环变量 $dist 和 $align 填入对应的参数位置
        python main.py \
            --dataset ur_fall \
            --global_rounds 200 \
            --local_epochs 2 \
            --numusers 0.5 \
            --optimizer Adam \
            --algo fedprop \
            --pfl \
            --manual "$manual_name" \
            --server_dist_weight $dist \
            --server_align_weight 0.00 \
            --client_orth_weight 0.1 \
            --client_align_weight $align \
            --client_reg_weight 0.00 \
            --client_logits_weight 0.5
        
        # 3. 检查上一步是否成功 (可选，防止报错了还继续跑)
        if [ $? -ne 0 ]; then
            echo "错误：实验 $manual_name 运行失败，脚本终止。"
            exit 1
        fi

        sleep 1 # 稍微暂停，让文件系统IO完成
    done
done

echo "所有 25 组实验运行完毕！"