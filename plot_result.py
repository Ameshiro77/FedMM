import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np  # 必须引入 numpy
from export import generate_table

# ================= 配置区域 =================

results_dir = "results"
font_path = "times.ttf"

# 数据集定义
datasets_modalities = {
    "mhealth": ["acce", "gyro", "mage"],
    "opp": ["acce", "gyro"],
    # "hapt": ["acce", "gyro"],
    "ur_fall": ["acce", "rgb", "depth"]
}

# [核心配置] 文件名(Key) -> 显示名称(Value)
file_method_map = {
    "fedprop":   "FedProp",
    "fedcent":   "Single",
    "fedavg":    "FedAvg",
    "fedprox":   "FedProx",
    "fedproto":  "FedProto",
    "fedmekt":   "FedMEKT",
    "feddtg":    "FedDTG",
}

# [颜色配置]
method_colors = {
    "FedProp":  "red",
    "Single":   "gray",
    "FedAvg":   "blue",
    "FedProx":  "cyan",
    "FedProto": "green",
    "FedMEKT":  "orange",
    "FedDTG":   "#FF00FF",
}

# ================= 论文绘图风格设置 =================

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
else:
    print(f"[警告] 未找到字体文件: {font_path}，使用默认字体")

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'lines.linewidth': 2.5
})

# ================= 核心逻辑 =================

def generate_combined_paper_plot(results_dir="results"):
    dataset_names = list(datasets_modalities.keys())
    num_datasets = len(dataset_names)
    
    print(f"正在生成组合图，目标目录: {results_dir} ...")

    # 创建画布
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 6))
    if num_datasets == 1: axes = [axes]

    # 遍历数据集
    for i, dataset in enumerate(dataset_names):
        ax = axes[i]
        dataset_dir = os.path.join(results_dir, dataset.lower())
        
        ax.set_title(dataset.upper().replace("_", "-")) 
        
        # 遍历算法
        for file_key, display_name in file_method_map.items():
            # 1. 严格按照你的文件名格式 {file_key}_acc.json
            json_file = os.path.join(dataset_dir, f"{file_key}_acc.json")
            
            # 兼容大小写
            target_file = json_file
            if not os.path.exists(target_file) and os.path.exists(json_file.lower()):
                target_file = json_file.lower()

            if os.path.exists(target_file):
                try:
                    with open(target_file, 'r') as f:
                        data = json.load(f)
                        
                        # 2. 读取精度数据 (优先 global, 其次兼容 PFL 的 local)
                        global_acc = data.get("global_accuracy", [])
                        
                        if not global_acc:
                            # 尝试查找其他可能的键名
                            for k in ["mean_local_accuracy", "test_accuracy", "accuracy", "acc"]:
                                if k in data and len(data[k]) > 0:
                                    global_acc = data[k]
                                    break
                        
                        # 如果还没找到数据，跳过
                        if not global_acc:
                            continue

                        rounds = data.get("rounds", list(range(1, len(global_acc) + 1)))
                        color = method_colors.get(display_name, None)
                        
                        # === 3. 针对 ur_fall 的平滑逻辑 ===
                        if dataset == "ur_fall" and len(global_acc) > 10:
                            window_size = 2  # 建议设为 10，3 有点太小了看不出效果
                            kernel = np.ones(window_size) / window_size
                            
                            # 平滑处理
                            global_acc = np.convolve(global_acc, kernel, mode='valid')
                            # 对齐 X 轴
                            rounds = rounds[window_size-1:]
                        # ================================

                        ax.plot(rounds, global_acc, label=display_name, color=color)
                        
                except Exception as e:
                    print(f"读取错误 {target_file}: {e}")
        
        ax.set_xlabel("Communication Rounds")
        if i == 0: ax.set_ylabel("Test Accuracy")
        ax.grid(True, linestyle='--', alpha=0.6)

    # === 生成顶部共享图例 ===
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # 去重

    n_cols = len(by_label)
        
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
               bbox_to_anchor=(0.5, 0.95),
               ncol=n_cols, frameon=False)

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98, wspace=0.2)

    # 保存文件
    output_path = os.path.join(results_dir, "comparisons", "combined_accuracy_paper.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"PDF 已保存至: {output_path}")

if __name__ == "__main__":
    # 处理 Global Results
    generate_combined_paper_plot("results")

    # 处理 PFL Results
    generate_combined_paper_plot("results/pfl")