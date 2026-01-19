import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
# 假设您的 generate_table 在 utils.plot_util 中，如果不是请调整引用
try:
    from utils.plot_util import generate_table
except ImportError:
    # 兜底：如果找不到模块，定义一个空函数以免报错
    def generate_table(*args, **kwargs):
        print("[Warning] generate_table function not found.")

# ================= Configuration Area =================

results_dir = "results"
font_path = "times.ttf"

# Datasets and Modalities
datasets_modalities = {
    "mhealth": ["acce", "gyro", "mage"],
    "opp": ["acce", "gyro"],
    # "hapt": ["acce", "gyro"],
    "ur_fall": ["acce", "rgb", "depth"]
}

# [Core Config] Filename Prefix (Key) -> Display Name (Value)
file_method_map = {
    "fedcent":   "Single",
    "fedavg":    "FedAvg",
    # "fedprox":   "FedProx",
    "fedproto":  "FedProto",
    # "fedgao":    "FedPL",
    "feddtg":    "FedDTG",
    "fedmekt":   "FedMEKT",
    "fedprop":   "FedBiKD",
}

# [Color Config] Display Name -> Fixed Color
# [Color Config] Display Name -> Fixed Color
method_colors = {
    "Single":   "gray",
    "FedAvg":   "blue",
    "FedProx":  "cyan",
    "FedProto": "green",
    "FedMEKT":  "orange",
    "FDARN":    "purple",
    "FedMEMA":  "brown",
    
    # "FedProp":  "red",    # <--- 删除或修改这一行
    "FedBiKD":  "red",      # <--- 【修改】确保 Key 与 file_method_map 中的 Value 一致！
    
    "FedDTG":   "#FF00FF",
    "FedPL":    "olive",    
    "FedAB":    "pink",
    "FedPropGEN": "olive"
}

# ================= Paper Style Settings (From Script 1) =================

# 尝试加载 Times New Roman 字体
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
else:
    print(f"[警告] 未找到字体文件: {font_path}，使用默认 Serif 字体")

# 统一更新绘图参数
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'lines.linewidth': 2.5,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

# ================= Function 1: Detailed Comparison (Separate SVGs) =================

def compare_algorithms(results_dir="results"):
    """
    生成每个数据集详细的 Accuracy 和 F1 对比图 (SVG格式)
    """
    plot_data = {ds: {} for ds in datasets_modalities.keys()}
    print(f"\n[Task 1] Start generating comparison SVGs from {results_dir}...")

    # 1. Load Data
    for dataset in datasets_modalities.keys():
        dataset_dir = os.path.join(results_dir, dataset.lower())
        
        for file_key, display_name in file_method_map.items():
            json_file = os.path.join(dataset_dir, f"{file_key}_acc.json")
            
            # Case-insensitive fallback
            target_file = json_file
            if not os.path.exists(target_file) and os.path.exists(json_file.lower()):
                target_file = json_file.lower()

            if os.path.exists(target_file):
                try:
                    with open(target_file, 'r') as f:
                        data = json.load(f)
                        global_acc = data.get("global_accuracy", [])
                        global_f1 = data.get("global_f1", [])
                        rounds = data.get("rounds", list(range(1, len(global_acc) + 1)))

                        plot_data[dataset][file_key] = {
                            "accuracy": global_acc,
                            "f1_score": global_f1,
                            "rounds": rounds,
                            "label": display_name
                        }
                except Exception as e:
                    print(f"Error reading {target_file}: {e}")

    # 2. Plot
    comparison_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    for dataset, alg_data in plot_data.items():
        if not alg_data:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for file_key in file_method_map.keys():
            if file_key in alg_data:
                data = alg_data[file_key]
                rounds = data["rounds"]
                accuracy = data["accuracy"]
                f1_score = data["f1_score"]
                label_name = data["label"] 
                line_color = method_colors.get(label_name, None)
                
                ax1.plot(rounds, accuracy, label=label_name, linewidth=2, color=line_color)
                if f1_score and len(f1_score) > 0:
                    ax2.plot(rounds, f1_score, label=label_name, linewidth=2, color=line_color)

        ax1.set_xlabel("Communication Rounds")
        ax1.set_ylabel("Global Accuracy")
        ax1.set_title(f"Accuracy - {dataset.upper()}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Communication Rounds")
        ax2.set_ylabel("Global F1 Score")
        ax2.set_title(f"F1 Score - {dataset.upper()}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(comparison_dir, f"{dataset}_comparison.svg")
        plt.savefig(comparison_path, format="svg", bbox_inches='tight')
        plt.close()
        print(f"Saved SVG: {comparison_path}")

# ================= Function 2: Combined Paper Plot (Single PDF) =================

def generate_combined_paper_plot(results_dir="results"):
    """
    生成论文专用的组合长图 (所有数据集在一行, 统一图例, PDF格式)
    """
    dataset_names = list(datasets_modalities.keys())
    num_datasets = len(dataset_names)
    
    print(f"\n[Task 2] Generating Combined Paper PDF in {results_dir} ...")

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
            json_file = os.path.join(dataset_dir, f"{file_key}_acc.json")
            
            target_file = json_file
            if not os.path.exists(target_file) and os.path.exists(json_file.lower()):
                target_file = json_file.lower()

            if os.path.exists(target_file):
                try:
                    with open(target_file, 'r') as f:
                        data = json.load(f)
                        
                        # 优先读取 global, 其次尝试 local (兼容 PFL)
                        global_acc = data.get("global_accuracy", [])
                        if not global_acc:
                            for k in ["mean_local_accuracy", "test_accuracy", "accuracy", "acc"]:
                                if k in data and len(data[k]) > 0:
                                    global_acc = data[k]
                                    break
                        
                        if not global_acc:
                            continue

                        rounds = data.get("rounds", list(range(1, len(global_acc) + 1)))
                        color = method_colors.get(display_name, None)
                        
                        # === 针对 ur_fall 的平滑逻辑 ===
                        if dataset == "ur_fall" and len(global_acc) > 10:
                            window_size = 2 
                            kernel = np.ones(window_size) / window_size
                            global_acc = np.convolve(global_acc, kernel, mode='valid')
                            rounds = rounds[window_size-1:]
                        # ================================

                        ax.plot(rounds, global_acc, label=display_name, color=color)
                        
                except Exception as e:
                    print(f"Read error {target_file}: {e}")
        
        ax.set_xlabel("Communication Rounds")
        if i == 0: ax.set_ylabel("Test Accuracy")
        ax.grid(True, linestyle='--', alpha=0.6)

    # === 生成顶部共享图例 ===
    # 获取最后一个子图的句柄和标签，用于生成去重图例
    handles, labels = axes[-1].get_legend_handles_labels()
    
    # 按照 file_method_map 的顺序重新排序图例 (可选，为了美观)
    # 这里简单使用去重逻辑
    by_label = dict(zip(labels, handles)) 
    
    # 调整图例列数
    n_cols = len(by_label)
        
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
               bbox_to_anchor=(0.5, 0.92), # 调整到图表上方
               ncol=n_cols, frameon=False)

    # 调整布局留出图例空间
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98, wspace=0.2)

    # 保存文件
    output_path = os.path.join(results_dir, "comparisons", "combined_accuracy_paper.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved Combined PDF: {output_path}")

# ================= Main Execution =================

if __name__ == "__main__":
    
    # # --- 1. Process Global Results (Default) ---
    # print("="*40)
    # print("Processing GLOBAL Results")
    # print("="*40)
    # compare_algorithms(results_dir="results")
    # generate_table(results_dir="results", file_method_map=file_method_map)
    # generate_combined_paper_plot(results_dir="results")

    # --- 2. Process PFL Results ---
    print("\n" + "="*40)
    print("Processing PFL Results")
    print("="*40)
    compare_algorithms(results_dir="results/pfl")
    generate_table(results_dir="results/pfl", file_method_map=file_method_map)
    generate_combined_paper_plot(results_dir="results/pfl")