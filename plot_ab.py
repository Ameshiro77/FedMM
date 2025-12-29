import os,math
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np  # 【修改1】引入 numpy 用于计算平滑

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

import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

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


def plot_radar_chart(dataset, dist_results, save_dir):
    """
    绘制超参数对比雷达图 (大字体 + LaTeX 标签版)
    """
    # === 1. 数据提取 ===
    labels = []
    server_accs = []
    client_accs = []

    sorted_keys = sorted(dist_results.keys())
    
    for key in sorted_keys:
        rec = dist_results[key]
        s_acc = rec["server_acc_curve"][-1] if rec["server_acc_curve"] else 0.0
        c_acc = rec.get("client_avg_acc", 0.0)
        
        # === 【核心修改点】 生成 LaTeX 格式标签 ===
        # 1. 提取纯数值字符串 (例如 "0.1")
        value_str = key.replace(f"{dataset}_", "").replace("fedprop_serverdist_", "").replace(".json", "")
        
        # 2. 包装成 LaTeX 格式
        # 注意：f-string 中使用 LaTeX 的花括号需要双写 {{ }} 进行转义
        # r 表示 raw string，处理反斜杠
        tex_label = fr"$\lambda^S_{{dist}}={value_str}$"
        labels.append(tex_label)
        # ========================================

        server_accs.append(s_acc)
        client_accs.append(c_acc)

    # 数据准备
    plot_s_accs = server_accs
    plot_c_accs = client_accs
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 闭合圆环
    plot_s_accs = np.concatenate((plot_s_accs, [plot_s_accs[0]]))
    plot_c_accs = np.concatenate((plot_c_accs, [plot_c_accs[0]]))
    angles += [angles[0]]
    labels += [labels[0]]

    # === 绘图设置 (保持大字体优化) ===
    # 适当增加画布宽度以容纳更长的标签
    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw=dict(polar=True))

    # Server Acc
    ax.plot(angles, plot_s_accs, color='#1f77b4', linewidth=3, linestyle='-', marker='o', markersize=8, label='Server Acc')
    ax.fill(angles, plot_s_accs, color='#1f77b4', alpha=0.15)

    # Client Avg Acc
    ax.plot(angles, plot_c_accs, color='#ff7f0e', linewidth=3, linestyle='--', marker='s', markersize=8, label='Client Avg Acc')
    ax.fill(angles, plot_c_accs, color='#ff7f0e', alpha=0.15)

    # 设置外圈标签 (使用 LaTeX 格式)
    # fontsize 稍微调小一点点以防标签过长重叠，或者增加 padding
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1], fontsize=16)
    # 增加标签与图表的距离 (padding)
    ax.tick_params(axis='x', pad=20)

    # 设置径向刻度字号
    ax.tick_params(axis='y', labelsize=14) 
    
    # 标题和图例
    # ax.set_title(f"Hyper-parameter Trade-off on {dataset}", fontsize=22, weight='bold', y=1.18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=16, frameon=True)
    
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)

    save_path = os.path.join(save_dir, f"{dataset}_dist_radar.pdf")
    
    # 使用 tight_layout 并增加一些 padding 确保长标签不被裁切
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Radar Chart (LaTeX Labels) => {save_path}")

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
        plot_radar_chart(
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
        # ===== non-PFL =====
        try:
            run_one(
                dataset=args.dataset,
                exp_type=exp_type,
                pfl=False
            )
        except FileNotFoundError as e:
            print(f"[Skip] {exp_type} (non-PFL): {e}")

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