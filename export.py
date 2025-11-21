import json
import os

results_dir = "results"
output_dir = os.path.join(results_dir, "comparisons")
os.makedirs(output_dir, exist_ok=True)

# 数据集及其模态
datasets_modalities = {
    "mhealth": ["acce", "gyro", "mage"],
    "opp": ["acce", "gyro"],
    "hapt": ["acce", "gyro"]
}

# 方法列表，FedCent 替代 Single
methods = ["FedCent", "FedAvg", "Fedprox", "Fedproto", "FedMEKT", "FDARN", "FedMEMA", "FedProp"]

# 方法对应的文件名
file_method_map = {
    "FedCent": "fedcent",
    "FedAvg": "fedavg",
    "Fedprox": "fedprox",
    "Fedproto": "fedproto",
    "FedMEKT": "fedmekt",
    "FDARN": "fdarn",
    "FedMEMA": "fedmema",
    "FedProp": "fedprop"
}

def load_data(dataset, algorithm):
    """读取服务端和客户端精度"""
    dataset_dir = os.path.join(results_dir, dataset.lower())
    acc_file = os.path.join(dataset_dir, f"{file_method_map[algorithm]}_acc.json")
    clients_file = os.path.join(dataset_dir, f"{file_method_map[algorithm]}_clients_accs.json")

    # 服务端精度
    server_acc = 0
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            data = json.load(f)
        global_acc = data.get("global_accuracy", [])
        if global_acc:
            server_acc = global_acc[-1] * 100  # 转成百分比

    # 客户端精度
    avg_client_acc = 0
    avg_modality_acc = {m: 0 for m in datasets_modalities[dataset]}
    if os.path.exists(clients_file):
        with open(clients_file, 'r') as f:
            data = json.load(f)
        avg_client_acc = data.get("avg_client_acc", 0) * 100
        for m in datasets_modalities[dataset]:
            avg_modality_acc[m] = data.get("avg_modality_acc", {}).get(m, 0) * 100

    return server_acc, avg_client_acc, avg_modality_acc

def generate_table():
    table_file = os.path.join(output_dir, "table_accuracy.txt")

    # 收集所有数据先计算每列最大值
    table_data = {method: {} for method in methods}
    for method in methods:
        for dataset in datasets_modalities:
            server_acc, avg_client_acc, avg_modality_acc = load_data(dataset, method)
            table_data[method][dataset] = {
                "modality": avg_modality_acc,
                "client": avg_client_acc,
                "server": server_acc
            }

    # 计算每列最大值
    col_max = {}
    for dataset, modalities in datasets_modalities.items():
        for m in modalities + ["client", "server"]:
            col_max[(dataset, m)] = max(
                table_data[method][dataset]["modality"].get(m, table_data[method][dataset].get(m))
                if m in modalities else table_data[method][dataset][m]
                for method in methods
            )

    # 写入表格文件
    with open(table_file, 'w') as f:
        for method in methods:
            line = method
            for dataset, modalities in datasets_modalities.items():
                avg_modality_acc = table_data[method][dataset]["modality"]
                avg_client_acc = table_data[method][dataset]["client"]
                server_acc = table_data[method][dataset]["server"]
                # 每个模态列
                for m in modalities:
                    val = avg_modality_acc[m]
                    val_str = f"{val:.2f}"
                    if val == col_max[(dataset, m)]:
                        line += f" & \\textbf{{{val_str}}}"
                    else:
                        line += f" & {val_str}"
                # 客户端平均
                val = avg_client_acc
                val_str = f"{val:.2f}"
                if val == col_max[(dataset, "client")]:
                    line += f" & \\textbf{{{val_str}}}"
                else:
                    line += f" & {val_str}"
                # 服务端
                val = server_acc
                val_str = f"{val:.2f}"
                if val == col_max[(dataset, "server")]:
                    line += f" & \\textbf{{{val_str}}}"
                else:
                    line += f" & {val_str}"
            line += " \\\\\n"
            f.write(line)

    print(f"Accuracy table saved to {table_file}")

generate_table()
