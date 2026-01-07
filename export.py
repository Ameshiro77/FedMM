import json
import os
import matplotlib.pyplot as plt

# 数据集及其模态
datasets_modalities = {
    "mhealth": ["acce", "gyro", "mage"],
    "opp": ["acce", "gyro"],
    "ur_fall": ["acce", "rgb", "depth"]
}

# 文件名(Key) -> 显示名称(Value)
file_method_map = {
    "fedcent":  "Single",
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedproto": "FedProto",
    # "fedab":    "FedAB",
    # "fedpropgen": "FedPropGEN",
    "feddtg":   "FedDTG",
    "fedmekt":  "FedMEKT",
    "fedprop":  "FedProp",
}

def load_data(dataset, file_prefix, results_dir):
    """
    读取服务端和客户端精度
    """
    dataset_dir = os.path.join(results_dir, dataset.lower())
    
    acc_file = os.path.join(dataset_dir, f"{file_prefix}_acc.json")
    clients_file = os.path.join(dataset_dir, f"{file_prefix}_clients_accs.json")

    # --- 服务端 ---
    server_acc = 0
    if os.path.exists(acc_file):
        try:
            with open(acc_file, 'r') as f:
                data = json.load(f)
            global_acc = data.get("global_accuracy", [])
            if global_acc:
                server_acc = global_acc[-1] * 100
        except Exception:
            pass

    # --- 客户端 ---
    avg_client_acc = 0
    avg_modality_acc = {m: 0 for m in datasets_modalities[dataset]}
    
    if os.path.exists(clients_file):
        try:
            with open(clients_file, 'r') as f:
                data = json.load(f)

            avg_client_acc = data.get("avg_client_acc", 0) * 100
            raw_modality_data = data.get("avg_modality_acc", {})
            modality_data_lower = {k.lower(): v for k, v in raw_modality_data.items()}

            for m in datasets_modalities[dataset]:
                if m.lower() in modality_data_lower:
                    avg_modality_acc[m] = modality_data_lower[m.lower()] * 100
        except Exception as e:
            print(f"Error reading {clients_file}: {e}")

    return server_acc, avg_client_acc, avg_modality_acc


def generate_table(results_dir="results"):
    output_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    table_file = os.path.join(output_dir, "table_accuracy.txt")
    active_keys = list(file_method_map.keys())

    table_data = {k: {} for k in active_keys}

    for key in active_keys:
        for dataset in datasets_modalities:
            server_acc, avg_client_acc, avg_modality_acc = load_data(
                dataset, key, results_dir
            )
            table_data[key][dataset] = {
                "modality": avg_modality_acc,
                "client": avg_client_acc,
                "server": server_acc
            }

    # 计算最大值（加粗用）
    col_max = {}
    for dataset, modalities in datasets_modalities.items():
        for m in modalities + ["client", "server"]:
            values = []
            for key in active_keys:
                if m in modalities:
                    val = table_data[key][dataset]["modality"].get(m, 0)
                elif m == "client":
                    val = table_data[key][dataset]["client"]
                else:
                    val = table_data[key][dataset]["server"]
                values.append(val)
            col_max[(dataset, m)] = max(values) if values else 0

    # 写表格
    with open(table_file, 'w') as f:
        for key in active_keys:
            line = file_method_map[key]
            for dataset, modalities in datasets_modalities.items():
                d = table_data[key][dataset]

                for m in modalities:
                    v = d["modality"][m]
                    s = f"{v:.2f}"
                    line += f" & \\textbf{{{s}}}" if v == col_max[(dataset, m)] and v > 0 else f" & {s}"

                for tag in ["client", "server"]:
                    v = d[tag]
                    s = f"{v:.2f}"
                    line += f" & \\textbf{{{s}}}" if v == col_max[(dataset, tag)] and v > 0 else f" & {s}"

            f.write(line + " \\\\\n")

    print(f"Accuracy table saved to {table_file}")


if __name__ == "__main__":
    generate_table("results")
    generate_table("results/pfl")
