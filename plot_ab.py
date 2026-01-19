import os,math
import json
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np  # 【修改1】引入 numpy 用于计算平滑

font_path = "times.ttf"
fm.fontManager.addfont(font_path)
def load_experiment_results(dataset, exp_type="ablation", pfl=False):
    """
    从指定文件夹加载 json：
    results/(pfl/){exp_type}/*.json
    """
    assert exp_type in ["ablation", "hyper_dist", "hyper_align"]

    if pfl:
        base_dir = f"./results/pfl/{exp_type}"
    else:
        base_dir = f"./results/{exp_type}"

    pattern = os.path.join(base_dir, f"{dataset}_*.json")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No json found in {base_dir} for dataset={dataset}"
        )

    results = {}
    for fp in files:
        with open(fp, "r") as f:
            rec = json.load(f)

        scheme = rec.get("scheme", os.path.basename(fp))
        results[scheme] = rec

    return results, base_dir


# ================= 全局字体设置 =================
# 为了让 LaTeX 公式和整体图表看起来更像论文出版级
# 将字体设置为 Serif (通常会映射到 Times New Roman 或类似字体)
plt.rcParams['font.family'] = 'serif'
# 如果你想强制指定 Times New Roman，取消下面这行的注释(前提是系统安装了该字体)
# plt.rcParams['font.serif'] = ['Times New Roman']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# ==============================================

# ... (load_experiment_results 函数保持不变，此处省略) ...
# ... (plot_server_acc 函数保持不变，此处省略) ...
# ... (save_ablation_table 函数保持不变，此处省略) ...

def plot_hyper_dist_line_chart(dataset, dist_results, save_dir):
    """
    【新函数】绘制超参数对比折线图 (横坐标: 参数值, 纵坐标: 精度)
    替代原来的雷达图，展示 lambda_dist 对 Server/Client 准确率的影响。
    """
    # === 1. 数据提取与解析 ===
    param_values = []
    server_accs = []
    client_accs = []

    # 按文件名排序，确保折线顺序正确
    sorted_keys = sorted(dist_results.keys())

    for key in sorted_keys:
        rec = dist_results[key]
        # 获取最终精度
        s_acc = rec["server_acc_curve"][-1] if rec["server_acc_curve"] else 0.0
        c_acc = rec.get("client_avg_acc", 0.0)
        
        # 从 key 中提取参数值 (假设 key 格式类似 "dataset_fedprop_serverdist_0.1")
        # 您之前的逻辑是: value_str = key.replace(...).replace(...)
        # 这里为了稳健，尝试提取数字，如果提取不到则保留原字符串
        try:
            # 尝试提取浮点数
            val_str = key.replace(f"{dataset}_", "").replace("fedprop_serverdist_", "").replace(".json", "")
            val = float(val_str)
        except ValueError:
            val = 0.0 # 兜底，或根据实际情况调整
            
        param_values.append(val)
        server_accs.append(s_acc)
        client_accs.append(c_acc)

    # === 2. 绘图设置 ===
    plt.figure(figsize=(8, 6)) # 调整为适合折线图的比例

    # 字体设置 (沿用您之前的逻辑)
    font_path = 'times.ttf'
    try:
        my_font = fm.FontProperties(fname=font_path, size=18)
    except:
        my_font = fm.FontProperties(size=18)
    
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # === 3. 绘制两条折线 ===
    # Line 1: Server Accuracy
    plt.plot(param_values, server_accs, 
             color='#1f77b4',       # 蓝色
             marker='o',            # 圆点标记
             markersize=8,          # 标记大小
             linewidth=2.5,         # 线宽
             linestyle='-',         # 实线
             label='Server Accuracy')

    # Line 2: Client Average Accuracy
    plt.plot(param_values, client_accs, 
             color='#ff7f0e',       # 橙色
             marker='s',            # 方块标记
             markersize=8, 
             linewidth=2.5, 
             linestyle='--',        # 虚线
             label='Avg. Client Accuracy')

    # === 4. 坐标轴与标签 ===
    plt.xlabel(r"Hyperparameter $\lambda^S_{dist}$", fontproperties=my_font)
    plt.ylabel("Accuracy", fontproperties=my_font)
    
    # 设置刻度字体
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(my_font)
        
    # 如果横坐标是离散的特定值，强制显示这些刻度
    plt.xticks(param_values, [str(v) for v in param_values])

    # === 5. 图例与网格 ===
    plt.legend(prop=my_font, loc='best', frameon=True, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存
    save_path = os.path.join(save_dir, f"{dataset}_dist_line.pdf")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Hyperparam Line Chart => {save_path}")

def plot_radar_chart(dataset, dist_results, save_dir):
    """
    绘制超参数对比雷达图 
    (刻度逆时针旋转10度 + 9x8.5尺寸 + 小图例)
    """
    # === 0. 字体强制设置 ===
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix' 
    
    # === 1. 数据提取 (保持不变) ===
    labels = []
    server_accs = []
    client_accs = []

    sorted_keys = sorted(dist_results.keys())
    
    for key in sorted_keys:
        rec = dist_results[key]
        s_acc = rec["server_acc_curve"][-1] if rec["server_acc_curve"] else 0.0
        c_acc = rec.get("client_avg_acc", 0.0)
        
        value_str = key.replace(f"{dataset}_", "").replace("fedprop_serverdist_", "").replace(".json", "")
        tex_label = fr"$\lambda^S_{{dist}}={value_str}$"
        labels.append(tex_label)

        server_accs.append(s_acc)
        client_accs.append(c_acc)

    # 闭合圆环
    plot_s_accs = server_accs
    plot_c_accs = client_accs
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    plot_s_accs = np.concatenate((plot_s_accs, [plot_s_accs[0]]))
    plot_c_accs = np.concatenate((plot_c_accs, [plot_c_accs[0]]))
    angles += [angles[0]]
    labels += [labels[0]]

    # === 2. 绘图设置 (9x8.5) ===
    fig, ax = plt.subplots(figsize=(9, 8.5), subplot_kw=dict(polar=True))

    # --- 绘制线条 ---
    ax.plot(angles, plot_s_accs, color='#1f77b4', linewidth=5, linestyle='-', 
            marker='o', markersize=12, label='Server Accuracy')
    ax.fill(angles, plot_s_accs, color='#1f77b4', alpha=0.15)

    ax.plot(angles, plot_c_accs, color='#ff7f0e', linewidth=5, linestyle='--', 
            marker='s', markersize=12, label='Average Client Accuracy')
    ax.fill(angles, plot_c_accs, color='#ff7f0e', alpha=0.15)

    # === 3. 字体与刻度设置 ===
    
    # 3.1 外圈标签
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1], fontsize=28)
    ax.tick_params(axis='x', pad=15) 

    # 3.2 径向刻度 (中间的数字)
    # 【核心修改】：逆时针旋转 10 度 (Matplotlib 极坐标默认逆时针为正)
    # 这样数字就会从正右方(0度)往上挪一点，避开轴线
    ax.set_rlabel_position(10)  
    
    # 恢复正常的字号设置，去掉之前的 zorder/bbox hack
    ax.tick_params(axis='y', labelsize=22) 

    # --- 4. 图例设置 (小字体 18) ---
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15), fontsize=18, frameon=True)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=2.5)

    save_path = os.path.join(save_dir, f"{dataset}_dist_radar.pdf")
    
    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Radar Chart (Rotated rlabels 10 deg) => {save_path}")

# ... (run_one 和 main 函数保持不变，此处省略) ...

import matplotlib.font_manager as fm  # 引入字体管理模块

def plot_server_acc(dataset, ablation_results, save_dir, smooth_window=1, exp_type="ablation"):
    plt.figure(figsize=(8, 5))

    # ================= 字体设置 (基于 times.ttf) =================
    # 1. 加载本地字体文件
    font_path = 'times.ttf'  # 请确保该文件存在，或修改为绝对路径
    try:
        # 创建字体属性对象，字号设为 16
        my_font = fm.FontProperties(fname=font_path, size=16)
        print(f"[Info] Loaded font from {font_path}")
    except:
        # 兜底：如果找不到文件，回退到默认
        print(f"[Warning] {font_path} not found, using default font.")
        my_font = fm.FontProperties(size=16)

    # 2. 配合 LaTeX 公式的设置 (公式字体尽量接近 Times)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    # ==========================================================

    # === LaTeX 图例映射 ===
    target_map = {
        "fedprop": "FedProp",
        "wo_server_dist": r"FedProp w/o $\mathcal{L}_{\mathrm{dist}}^S$",
        "wo_client_align": r"FedProp w/o $\mathcal{L}_{\mathrm{align}}^C$",
        "wo_client_dist": r"FedProp w/o $\mathcal{L}_{\mathrm{KD}}^C$"
    }

    has_plot = False

    for scheme, rec in ablation_results.items():
        acc = rec["server_acc_curve"]
        label_name = scheme

        if exp_type == "ablation":
            found = False
            for key, display_name in target_map.items():
                if key in scheme: 
                    label_name = display_name
                    found = True
                    break
            if not found:
                continue
        else:
            label_name = scheme.replace(f"{dataset}_", "").replace(".json", "")

        if smooth_window > 1 and len(acc) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            acc = np.convolve(acc, kernel, mode='valid')
        
        plt.plot(acc, label=label_name, linewidth=2)
        has_plot = True

    # === 【关键修改】将 fontproperties=my_font 应用到各个组件 ===
    
    # 1. 坐标轴标签
    plt.xlabel("Global Round", fontproperties=my_font)
    plt.ylabel("Server Accuracy", fontproperties=my_font)

    # 2. 图例 (注意参数名是 prop)
    if has_plot:
        plt.legend(prop=my_font)
    
    # 3. 刻度标签 (需要获取当前坐标轴对象 ax)
    ax = plt.gca()
    
    # 设置 X 轴刻度字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(my_font)
        
    # 设置 Y 轴刻度字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(my_font)

    plt.grid(True, linestyle='--', alpha=0.6)

    suffix = "_smooth" if smooth_window > 1 else ""
    save_path = os.path.join(save_dir, f"{dataset}_{exp_type}_server_acc{suffix}.pdf")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Plot ({exp_type}) => {save_path}")
def save_ablation_table(dataset, ablation_results, save_dir):
    save_path = os.path.join(save_dir, f"{dataset}_ablation.txt")

    # 取 modality 顺序
    first = next(iter(ablation_results.values()))
    modalities = list(first["modality_acc"].keys())

    # 收集数值（全部用百分制）
    table = {}
    for scheme, rec in ablation_results.items():
        server_final = (
            rec["server_acc_curve"][-1]
            if rec["server_acc_curve"]
            else 0.0
        )
        table[scheme] = {
            "modality": {m: rec["modality_acc"].get(m, 0.0) * 100 for m in modalities},
            "client": rec["client_avg_acc"] * 100,
            "server": server_final * 100
        }


    # 计算列最大值（用于 \\textbf）
    col_max = {}
    for m in modalities:
        col_max[m] = max(table[s]["modality"][m] for s in table)
    col_max["client"] = max(table[s]["client"] for s in table)
    col_max["server"] = max(table[s]["server"] for s in table)

    # 写 LaTeX 行
    with open(save_path, "w") as f:
        for scheme, d in table.items():
            # scheme 名字你可以在这里手动 map 成论文里的名字
            line = scheme

            # modality 列
            for m in modalities:
                v = d["modality"][m]
                s = f"{v:.2f}"
                if abs(v - col_max[m]) < 1e-6 and v > 0:
                    line += f" & \\textbf{{{s}}}"
                else:
                    line += f" & {s}"

            # client / server
            for tag in ["client", "server"]:
                v = d[tag]
                s = f"{v:.2f}"
                if abs(v - col_max[tag]) < 1e-6 and v > 0:
                    line += f" & \\textbf{{{s}}}"
                else:
                    line += f" & {s}"

            line += " \\\\\n"
            f.write(line)

    print(f"[Saved] LaTeX ablation table => {save_path}")


def run_one(dataset, exp_type, pfl):
    # 1. 加载所有数据
    results, save_dir = load_experiment_results(
        dataset=dataset,
        exp_type=exp_type,
        pfl=pfl
    )

    # 2. 根据实验类型选择绘图方式
    if exp_type == "hyper_dist":
        # === 新增逻辑：画雷达图 ===
        print(f"正在为 {exp_type} 生成雷达图...")
        plot_hyper_dist_line_chart(
            dataset=dataset,
            dist_results=results,
            save_dir=save_dir
        )
    else:
        # === 原有逻辑：画 Acc 曲线图 (ablation 或 hyper_align) ===
        current_smooth_window = 1 if dataset == "ur_fall" else 1
        
        plot_server_acc(
            dataset=dataset,
            ablation_results=results,
            save_dir=save_dir,
            smooth_window=current_smooth_window,
            exp_type=exp_type
        )

    # 3. 保存表格 (所有实验都生成表格)
    save_ablation_table(
        dataset=dataset,
        ablation_results=results,
        save_dir=save_dir
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    for exp_type in ["ablation", "hyper_dist", "hyper_align"]:
        # # ===== non-PFL =====
        # try:
        #     run_one(
        #         dataset=args.dataset,
        #         exp_type=exp_type,
        #         pfl=False
        #     )
        # except FileNotFoundError as e:
        #     print(f"[Skip] {exp_type} (non-PFL): {e}")

        # ===== PFL =====
        try:
            run_one(
                dataset=args.dataset,
                exp_type=exp_type,
                pfl=True
            )
        except FileNotFoundError as e:
            print(f"[Skip] {exp_type} (PFL): {e}")

if __name__ == "__main__":
    main()