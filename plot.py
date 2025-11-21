import json
import os
import matplotlib.pyplot as plt
from export import generate_table
def compare_algorithms(results_dir="results"):
    """
    绘制不同算法的精度和F1分数比较图
    """
    import glob
    
    # 查找所有算法的精度 JSON 文件
    pattern = os.path.join(results_dir, "**", "*_acc.json")
    json_files = glob.glob(pattern, recursive=True)
    
    if not json_files:
        print(f"No accuracy JSON files found in {results_dir}")
        return
    
    # 读取所有算法的数据
    algorithms_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            algorithm_name = data.get("algorithm", "Unknown")
            dataset_name = data.get("dataset", "Unknown")
            global_acc = data.get("global_accuracy", [])
            global_f1 = data.get("global_f1", [])  # 新增F1分数
            rounds = data.get("rounds", list(range(1, len(global_acc) + 1)))
            
            # 按数据集分组
            if dataset_name not in algorithms_data:
                algorithms_data[dataset_name] = {}
            
            algorithms_data[dataset_name][algorithm_name] = {
                "accuracy": global_acc,
                "f1_score": global_f1,  # 保存F1分数
                "rounds": rounds
            }
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # 为每个数据集绘制比较图
    for dataset, alg_data in algorithms_data.items():
        # 创建包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绘制精度比较图
        for alg_name, data in alg_data.items():
            rounds = data["rounds"]
            accuracy = data["accuracy"]
            ax1.plot(rounds, accuracy, label=alg_name, linewidth=2)
        
        ax1.set_xlabel("Communication Rounds")
        ax1.set_ylabel("Global Accuracy")
        ax1.set_title(f"Accuracy Comparison - {dataset}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制F1分数比较图
        for alg_name, data in alg_data.items():
            rounds = data["rounds"]
            f1_score = data["f1_score"]
            ax2.plot(rounds, f1_score, label=alg_name, linewidth=2)
        
        ax2.set_xlabel("Communication Rounds")
        ax2.set_ylabel("Global F1 Score")
        ax2.set_title(f"F1 Score Comparison - {dataset}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存比较图
        comparison_dir = os.path.join(results_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_path = os.path.join(comparison_dir, f"{dataset}_comparison.svg")
        plt.savefig(comparison_path, format="svg", bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {comparison_path}")
        
        # 在控制台输出最终精度和F1分数
        print(f"\nFinal Performance for {dataset}:")
        for alg_name, data in alg_data.items():
            final_acc = data["accuracy"][-1] if data["accuracy"] else 0
            final_f1 = data["f1_score"][-1] if data["f1_score"] else 0
            print(f"  {alg_name}: Accuracy={final_acc:.4f}, F1 Score={final_f1:.4f}")




compare_algorithms()
generate_table()