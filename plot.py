import json
import os
import matplotlib.pyplot as plt
from export import generate_table

# ================= Configuration Area =================

results_dir = "results"

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
    "fedprox":   "FedProx",
    "fedproto":  "FedProto",
    "fedmekt":   "FedMEKT",
    # "fdarn":     "FDARN",
    # "fedmema":   "FedMEMA",
    "fedprop":   "FedProp",
    # "fedab":     "FedAB",
    # "fedpropgen": "FedPropGEN",
    "feddtg":    "FedDTG",
}

# [NEW] Display Name -> Fixed Color
# This ensures "FedAvg" is always Blue, "FedProp" is always Red, etc.
method_colors = {
    "Single":   "gray",
    "FedAvg":   "blue",
    "FedProx":  "cyan",
    "FedProto": "green",
    "FedMEKT":  "orange",
    "FDARN":    "purple",
    "FedMEMA":  "brown",
    "FedProp":  "red",      # Highlight your method
    "FedDTG":   "#FF00FF",  # Magenta (Highlight your method)
    "FedAB":    "pink",
    "FedPropGEN": "olive"
}

# ===============================================================

def compare_algorithms(results_dir="results"):
    """
    Draw comparison plots based on configuration with consistent colors.
    """
    # Prepare data container
    plot_data = {ds: {} for ds in datasets_modalities.keys()}

    print(f"Start loading data from {results_dir}...")

    # 1. Iterate through datasets
    for dataset in datasets_modalities.keys():
        dataset_dir = os.path.join(results_dir, dataset.lower())
        
        # 2. Iterate through config
        for file_key, display_name in file_method_map.items():
            
            # Construct file path
            json_file = os.path.join(dataset_dir, f"{file_key}_acc.json")

            # Try reading (case-insensitive fallback)
            target_file = json_file
            if not os.path.exists(target_file) and os.path.exists(json_file.lower()):
                target_file = json_file.lower()

            if os.path.exists(target_file):
                try:
                    with open(target_file, 'r') as f:
                        data = json.load(f)
                        
                        # Get data
                        global_acc = data.get("global_accuracy", [])
                        global_f1 = data.get("global_f1", [])
                        rounds = data.get("rounds", list(range(1, len(global_acc) + 1)))

                        # Store in dictionary
                        plot_data[dataset][file_key] = {
                            "accuracy": global_acc,
                            "f1_score": global_f1,
                            "rounds": rounds,
                            "label": display_name # Store the display name
                        }
                except Exception as e:
                    print(f"Error reading {target_file}: {e}")
            else:
                pass 

    # 3. Plotting Logic
    comparison_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    for dataset, alg_data in plot_data.items():
        if not alg_data:
            print(f"No data found for dataset: {dataset}, skipping plot.")
            continue

        print(f"Plotting for {dataset}...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Iterate file_method_map again to ensure legend order matches config
        for file_key in file_method_map.keys():
            if file_key in alg_data:
                data = alg_data[file_key]
                rounds = data["rounds"]
                accuracy = data["accuracy"]
                f1_score = data["f1_score"]
                label_name = data["label"] 
                
                # [NEW] Get the fixed color for this algorithm
                # If not defined in method_colors, default to None (auto color)
                line_color = method_colors.get(label_name, None)
                
                # Plot Accuracy with specific color
                ax1.plot(rounds, accuracy, label=label_name, linewidth=2, color=line_color)
                
                # Plot F1 with specific color
                if f1_score and len(f1_score) > 0:
                    ax2.plot(rounds, f1_score, label=label_name, linewidth=2, color=line_color)

        # Settings for Accuracy Subplot
        ax1.set_xlabel("Communication Rounds")
        ax1.set_ylabel("Global Accuracy")
        ax1.set_title(f"Accuracy Comparison - {dataset}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Settings for F1 Subplot
        ax2.set_xlabel("Communication Rounds")
        ax2.set_ylabel("Global F1 Score")
        ax2.set_title(f"F1 Score Comparison - {dataset}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_path = os.path.join(comparison_dir, f"{dataset}_comparison.svg")
        plt.savefig(comparison_path, format="svg", bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {comparison_path}")

    print("\nComparison plots generation finished.")

if __name__ == "__main__":

    compare_algorithms(results_dir="results")
    generate_table(results_dir="results")

    # PFL
    compare_algorithms(results_dir="results/pfl")
    generate_table(results_dir="results/pfl")
