import os
import json
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import cm # 用于 3D 柱状图颜色映射

# ================= 全局配置 =================
TARGET_DIR = "results/pfl/manual"   # 目标文件夹路径
FONT_PATH = "times.ttf"             # 字体文件路径

# ================= 绘图风格设置 =================
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
else:
    print(f"[警告] 未找到 {FONT_PATH}，将使用系统默认 Serif 字体。")

# 统一图表样式
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,  # 3D图刻度密集，稍微调小一点
    'ytick.labelsize': 14,
    'legend.fontsize': 18,
    'lines.linewidth': 2.5,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

# 定义字体属性对象 (供局部使用)
font_prop = fm.FontProperties(fname=FONT_PATH, size=20) if os.path.exists(FONT_PATH) else None


# ==============================================================================
#  函数 1: Local Epochs 影响分析 (折线图)
# ==============================================================================

def process_epoch_analysis(data_dir):
    """
    负责处理 Local Epochs 实验：加载数据 -> 绘图 -> 保存
    """
    print(f"\n[任务 1] 开始处理 Local Epochs 分析...")
    save_name = "impact_of_local_epochs.pdf"
    
    # --- 1. 数据加载与解析 ---
    pattern = os.path.join(data_dir, "*ep*.json")
    files = glob.glob(pattern)
    
    # 过滤掉属于 Grid Search 的文件 (防止文件名撞车)
    files = [f for f in files if "sdist" not in os.path.basename(f)]

    if not files:
        print(f"  [跳过] 未找到 Epoch 相关数据")
        return

    extracted_data = []
    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(r"ep(\d+)", filename)
        if match:
            epoch_val = int(match.group(1))
            try:
                with open(file_path, "r") as f:
                    rec = json.load(f)
                s_acc = rec["server_acc_curve"][-1] if rec.get("server_acc_curve") else 0.0
                c_acc = rec.get("client_avg_acc", 0.0)
                extracted_data.append({"epoch": epoch_val, "server_acc": s_acc, "client_acc": c_acc})
            except Exception as e:
                print(f"  [错误] 读取 {filename}: {e}")

    if not extracted_data:
        return

    extracted_data.sort(key=lambda x: x["epoch"])

    # --- 2. 绘图 ---
    epochs = [d["epoch"] for d in extracted_data]
    server_accs = [d["server_acc"] for d in extracted_data]
    client_accs = [d["client_acc"] for d in extracted_data]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, server_accs, color='#1f77b4', marker='o', markersize=9, linestyle='-', label='Server Accuracy')
    plt.plot(epochs, client_accs, color='#ff7f0e', marker='s', markersize=9, linestyle='--', label='Avg. Client Accuracy')

    plt.xlabel("Local Epochs ($E$)", fontproperties=font_prop)
    plt.ylabel("Accuracy", fontproperties=font_prop)
    plt.xticks(epochs)
    plt.legend(frameon=True, edgecolor='gray', fancybox=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # --- 3. 保存 ---
    save_path = os.path.join(data_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [完成] Epoch 折线图已保存至: {save_path}")


# ==============================================================================
#  函数 2: 双超参数网格搜索分析 (包含 3D 柱状图 和 热力图)
# ==============================================================================

def process_grid_search_analysis(data_dir):
    """
    负责处理双超参数实验：
    1. 生成 3D 柱状图 (Server & Client)
    2. 生成 热力图 (Server & Client, 无数字)
    """
    print(f"\n[任务 2] 开始处理 Grid Search 分析...")

    # --- 1. 数据加载与解析 ---
    pattern = os.path.join(data_dir, "*sdist*calign*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"  [跳过] 未找到 Grid Search 相关数据")
        return

    data_records = []
    for fp in files:
        fname = os.path.basename(fp)
        match = re.search(r"sdist_([\d\.]+)_calign_([\d\.]+)", fname)
        if match:
            try:
                s_val, c_val = float(match.group(1)), float(match.group(2))
                with open(fp, 'r') as f:
                    rec = json.load(f)
                s_acc = rec["server_acc_curve"][-1] if rec.get("server_acc_curve") else 0.0
                c_acc = rec.get("client_avg_acc", 0.0)
                data_records.append({
                    "s_dist": s_val, "c_align": c_val,
                    "server_acc": s_acc, "client_acc": c_acc
                })
            except:
                pass

    if not data_records:
        return

    # --- 2. 构建矩阵 ---
    s_axis = sorted(list(set(d["s_dist"] for d in data_records))) # Y轴
    c_axis = sorted(list(set(d["c_align"] for d in data_records))) # X轴
    
    server_matrix = np.zeros((len(s_axis), len(c_axis)))
    client_matrix = np.zeros((len(s_axis), len(c_axis)))
    
    for d in data_records:
        r_idx = s_axis.index(d["s_dist"])
        c_idx = c_axis.index(d["c_align"])
        server_matrix[r_idx, c_idx] = d["server_acc"]
        client_matrix[r_idx, c_idx] = d["client_acc"]

    print(f"  检测到网格尺寸: {len(s_axis)}x{len(c_axis)}")

    # --------------------------------------------------------
    # 子任务 A: 绘制 3D 柱状图 (您要求的)
    # --------------------------------------------------------
    fig = plt.figure(figsize=(18, 8))
    
    # 准备 3D 坐标
    _x = np.arange(len(c_axis))
    _y = np.arange(len(s_axis))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    width, depth = 0.4, 0.4 # 柱子粗细

    # --- 左图: Server Accuracy (3D) ---
    ax1 = fig.add_subplot(121, projection='3d')
    top1 = server_matrix.ravel()
    bottom1 = np.zeros_like(top1)
    # 颜色映射
    colors1 = cm.viridis(top1 / np.max(top1) if np.max(top1) > 0 else top1)
    
    ax1.bar3d(x, y, bottom1, width, depth, top1, shade=True, color=colors1)
    ax1.set_title("Server Accuracy", fontproperties=font_prop, y=1.02)
    ax1.set_xlabel(r"$\lambda^C_{align}$", fontproperties=font_prop)
    ax1.set_ylabel(r"$\lambda^S_{dist}$", fontproperties=font_prop)
    ax1.set_zlabel("Accuracy", fontproperties=font_prop)
    # 设置刻度
    ax1.set_xticks(_x + width/2)
    ax1.set_xticklabels([str(v) for v in c_axis])
    ax1.set_yticks(_y + depth/2)
    ax1.set_yticklabels([str(v) for v in s_axis])
    ax1.view_init(elev=25, azim=-45)

    # --- 右图: Client Accuracy (3D) ---
    ax2 = fig.add_subplot(122, projection='3d')
    top2 = client_matrix.ravel()
    bottom2 = np.zeros_like(top2)
    colors2 = cm.plasma(top2 / np.max(top2) if np.max(top2) > 0 else top2)
    
    ax2.bar3d(x, y, bottom2, width, depth, top2, shade=True, color=colors2)
    ax2.set_title("Avg. Client Accuracy", fontproperties=font_prop, y=1.02)
    ax2.set_xlabel(r"$\lambda^C_{align}$", fontproperties=font_prop)
    ax2.set_ylabel(r"$\lambda^S_{dist}$", fontproperties=font_prop)
    ax2.set_zlabel("Accuracy", fontproperties=font_prop)
    
    ax2.set_xticks(_x + width/2)
    ax2.set_xticklabels([str(v) for v in c_axis])
    ax2.set_yticks(_y + depth/2)
    ax2.set_yticklabels([str(v) for v in s_axis])
    ax2.view_init(elev=25, azim=-45)

    plt.tight_layout()
    save_path_3d = os.path.join(data_dir, "hyperparams_grid_3d.pdf")
    plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [完成] 3D 柱状图已保存: {save_path_3d}")

    # --------------------------------------------------------
    # 子任务 B: 绘制 热力图 (不带数字)
    # --------------------------------------------------------
    fig_h, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(16, 7))
    
    def draw_heatmap(ax, matrix, title, cmap):
        im = ax.imshow(matrix, cmap=cmap, origin='lower', aspect='auto')
        ax.set_xticks(np.arange(len(c_axis)))
        ax.set_yticks(np.arange(len(s_axis)))
        ax.set_xticklabels([str(v) for v in c_axis], fontproperties=font_prop)
        ax.set_yticklabels([str(v) for v in s_axis], fontproperties=font_prop)
        ax.set_xlabel(r"Client Align Weight ($\lambda^C_{align}$)", fontproperties=font_prop)
        ax.set_ylabel(r"Server Dist Weight ($\lambda^S_{dist}$)", fontproperties=font_prop)
        ax.set_title(title, fontproperties=font_prop, y=1.02)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        # 不写 ax.text，保持干净

    draw_heatmap(ax_h1, server_matrix, "Server Accuracy Heatmap", 'viridis')
    draw_heatmap(ax_h2, client_matrix, "Avg. Client Accuracy Heatmap", 'plasma')

    plt.tight_layout()
    save_path_hm = os.path.join(data_dir, "hyperparams_grid_heatmap.pdf")
    plt.savefig(save_path_hm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [完成] 热力图已保存: {save_path_hm}")


# ==============================================================================
#  Main 直接调用
# ==============================================================================

def main():
    print(f"正在扫描数据目录: {TARGET_DIR}")
    
    # 1. 画 Epoch 影响图
    process_epoch_analysis(TARGET_DIR)
    
    # 2. 画 双超参数 Grid Search 图 (3D + 热力图)
    process_grid_search_analysis(TARGET_DIR)

if __name__ == "__main__":
    main()