import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import savemat, loadmat
from scipy.stats import zscore

data_path = "./data/"
PUBLIC_RATIO = 1
TRAIN_RATIO = 0.1
N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_DIV_HAPT = 100
N_LABEL_DIV_OPP = 30
N_LABEL_DIV_MHEALTH = 30
N_LABEL_DIV_URFALL = 30
N_LABEL_DIV_HAPT = 30

def load_data(dataset):
    data = dataset
    if data == "opp":
        mat_data = loadmat(os.path.join(data_path, "opp", "opp.mat"))
        
        modalities = ["acce", "gyro"]
         
        data_train = {m: zscore(mat_data[f"x_train_{m}"]) for m in modalities}
        data_train["y"] = np.squeeze(mat_data["y_train"])

        data_test = {m: zscore(mat_data[f"x_test_{m}"]) for m in modalities}
        data_test["y"] = np.squeeze(mat_data["y_test"])

        data_public = {m: zscore(mat_data[f"x_public_{m}"]) for m in modalities}
        data_public["y"] = np.squeeze(mat_data["y_public"])
        
        
        
        
        return (data_train, data_test, data_public)

    elif data == "mhealth":
        modalities = ["acce", "gyro", "mage"]
        mat_data = loadmat(os.path.join(data_path, "mhealth", "mhealth.mat"))

        s_test = np.random.randint(1, 11)
        s_public = np.random.choice([i for i in range(1, 11) if i != s_test])

        data_train = {m: [] for m in modalities}
        data_train["y"] = []

        data_test = {}
        data_public = {}

        for i in range(1, 11):
            subject_data = {m: zscore(mat_data[f"s{i}_{m}"]) for m in modalities}
            subject_label = np.squeeze(mat_data[f"s{i}_y"])

            if i == s_test:
                data_test = subject_data
                data_test["y"] = subject_label
            elif i == s_public:
                data_public = subject_data
                data_public["y"] = subject_label
            else:
                for m in modalities:
                    data_train[m].append(subject_data[m])
                data_train["y"].append(subject_label)

        # 拼接训练集
        for m in modalities:
            data_train[m] = np.concatenate(data_train[m])
        # data_train["y"] = np.squeeze(np.concatenate(data_train["y"], axis=1))
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        print(data_train['acce'].shape)
       
        return data_train, data_test, data_public
    
    elif data == "hapt":
        modalities = ["acce", "gyro"]
        mat_data = loadmat(os.path.join(data_path, "hapt", "hapt.mat"))

        all_subjects = list(range(1, 31))  # HAPT 共30个受试者
        s_test = np.random.choice(all_subjects, size=9, replace=False)  # 测试 9人
        remain = [i for i in all_subjects if i not in s_test]
        s_public = np.random.choice(remain, size=3, replace=False)       # 公共 3人
        s_train = [i for i in remain if i not in s_public]               # 剩余 18 训练

        print(f"Train: {len(s_train)} subjects, Test: {len(s_test)}, Public: {len(s_public)}")

        data_train = {m: [] for m in modalities}
        data_train["y"] = []
        data_test = {m: [] for m in modalities}
        data_test["y"] = []
        data_public = {m: [] for m in modalities}
        data_public["y"] = []

        for i in all_subjects:
            # 如果某个用户没有数据，跳过（保险）
            if f"s{i}_acce" not in mat_data:
                print(f"⚠️ Missing subject {i} in hapt.mat, skipped.")
                continue

            subject_data = {m: zscore(mat_data[f"s{i}_{m}"]) for m in modalities}
            subject_label = np.squeeze(mat_data[f"s{i}_y"])

            if i in s_test:
                for m in modalities:
                    data_test[m].append(subject_data[m])
                data_test["y"].append(subject_label)

            elif i in s_public:
                for m in modalities:
                    data_public[m].append(subject_data[m])
                data_public["y"].append(subject_label)

            else:
                for m in modalities:
                    data_train[m].append(subject_data[m])
                data_train["y"].append(subject_label)

        # 拼接
        for m in modalities:
            data_train[m] = np.concatenate(data_train[m])
            data_test[m] = np.concatenate(data_test[m])
            data_public[m] = np.concatenate(data_public[m])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
        data_public["y"] = np.squeeze(np.concatenate(data_public["y"]))
        # 将标签转为0-based（非常关键）
        for split in [data_train, data_test, data_public]:
            split["y"] = split["y"].astype(int) - 1

        print("✅ HAPT Data Split Done")
        print("Train shape:", data_train["acce"].shape)
        print("Test shape:", data_test["acce"].shape)
        print("Public shape:", data_public["acce"].shape)

    
        #  # ============================
        # # ✅ 数据完整性检查
        # # ============================
        # print("\n=== [HAPT 数据检查] ===")
        # for split_name, split_data in [("Train", data_train), ("Test", data_test), ("Public", data_public)]:
        #     print(f"\n--- {split_name} ---")
        #     for m in modalities:
        #         x = split_data[m]
        #         print(f"{m:>5s} shape: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}, NaN={np.isnan(x).any()}")

        #     y = split_data["y"]
        #     print(f"y shape: {y.shape}, unique={np.unique(y)}")
        #     print(f"y min={y.min()}, max={y.max()}, dtype={y.dtype}")
        #     # 检查是否有非整数标签
        #     if not np.allclose(y, y.astype(int)):
        #         print("⚠️ Warning: y contains non-integer labels!")

        #     # 检查是否需要减1（很多mat文件是1-based标签）
        #     if y.min() >= 1:
        #         print("⚠️ Labels are 1-based. Consider doing: y = y - 1")

        #     # 检查 NaN 或 Inf
        #     if np.isnan(y).any() or np.isinf(y).any():
        #         print("⚠️ Warning: NaN or Inf in labels!")

        # print("\n✅ 数据检查完成，准备返回。")
        # exit()

        return data_train, data_test, data_public


    elif data == "uci_har":
        modalities = ["acce", "gyro"]
        mat_data = loadmat(os.path.join(data_path, "uci_har", "uci_har.mat"))
        print(mat_data.keys())

        # 受试者编号范围：1~30
        all_subjects = list(range(1, 31))
        s_test = np.random.choice(all_subjects, size=9, replace=False)  # 9名测试
        remain = [i for i in all_subjects if i not in s_test]
        s_public = np.random.choice(remain, size=3, replace=False)       # 3名公共数据
        s_train = [i for i in remain if i not in s_public]               # 剩余 18 训练

        print(f"Train: {len(s_train)} subjects, Test: {len(s_test)}, Public: {len(s_public)}")

        data_train = {m: [] for m in modalities}
        data_train["y"] = []
        data_test = {m: [] for m in modalities}
        data_test["y"] = []
        data_public = {m: [] for m in modalities}
        data_public["y"] = []

        for i in all_subjects:
            subject_data = {m: zscore(mat_data[f"s{i}_{m}"]) for m in modalities}
            subject_label = np.squeeze(mat_data[f"s{i}_y"])

            if i in s_test:
                for m in modalities:
                    data_test[m].append(subject_data[m])
                data_test["y"].append(subject_label)

            elif i in s_public:
                for m in modalities:
                    data_public[m].append(subject_data[m])
                data_public["y"].append(subject_label)

            else:
                for m in modalities:
                    data_train[m].append(subject_data[m])
                data_train["y"].append(subject_label)

        # 拼接
        for m in modalities:
            data_train[m] = np.concatenate(data_train[m])
            data_test[m] = np.concatenate(data_test[m])
            data_public[m] = np.concatenate(data_public[m])
        data_train["acce"] = data_train["acce"].reshape(-1, 3)  # (6136*128, 3)
        data_train["gyro"] = data_train["gyro"].reshape(-1, 3)
        data_train["y"] = np.repeat(data_train["y"], 128)  # 标签同步展开

        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
        data_public["y"] = np.squeeze(np.concatenate(data_public["y"]))

        print("Train shape:", data_train["acce"].shape)
        print("Test shape:", data_test["acce"].shape)
        print("Public shape:", data_public["acce"].shape)
    
        return data_train, data_test, data_public

    elif data == "ur_fall":
        modalities = ["acce", "rgb", "depth"]
        mat_data = loadmat(os.path.join(data_path, "ur_fall", "ur_fall.mat"))

        # 随机选择 fall/ADL 测试集
        fall_test = np.random.choice(range(1, 31), 3, replace=False)
        adl_test = np.random.choice(range(1, 41), 4, replace=False)

        # 随机选择 fall/ADL 公共数据
        fall_public = np.random.choice([i for i in range(1, 31) if i not in fall_test], 3, replace=False)
        adl_public = np.random.choice([i for i in range(1, 41) if i not in adl_test], 4, replace=False)

        # 初始化训练集、测试集、公共数据
        data_train = {m: [] for m in modalities}
        data_train["y"] = []
        data_test = {}
        data_public = {}

        a_data = {m: mat_data[m] for m in modalities}
        a_y = mat_data["y"]

        # 处理 fall 数据
        for i in range(1, 31):
            subject_data = {}
            for m in modalities:
                sub_m = a_data[m][(a_data[m][:, 0] == 1) & (a_data[m][:, 1] == i), :]
                if m in ["acce", "depth"]:
                    sub_m[:, 3:] = zscore(sub_m[:, 3:])
                subject_data[m] = sub_m[:, 3:]
            subject_label = a_y[(a_y[:, 0] == 1) & (a_y[:, 1] == i), 3]

            if i in fall_test:
                data_test.update(subject_data)
                data_test["y"] = subject_label
            elif i in fall_public:
                data_public.update(subject_data)
                data_public["y"] = subject_label
            else:
                for m in modalities:
                    data_train[m].append(subject_data[m])
                data_train["y"].append(subject_label)

        # 处理 ADL 数据
        for i in range(1, 41):
            subject_data = {}
            for m in modalities:
                sub_m = a_data[m][(a_data[m][:, 0] == 0) & (a_data[m][:, 1] == i), :]
                if m in ["acce", "depth"]:
                    sub_m[:, 3:] = zscore(sub_m[:, 3:])
                subject_data[m] = sub_m[:, 3:]
            subject_label = a_y[(a_y[:, 0] == 0) & (a_y[:, 1] == i), 3]

            if i in adl_test:
                data_test.update(subject_data)
                data_test["y"] = subject_label
            elif i in adl_public:
                data_public.update(subject_data)
                data_public["y"] = subject_label
            else:
                for m in modalities:
                    data_train[m].append(subject_data[m])
                data_train["y"].append(subject_label)

        # 拼接训练集
        for m in modalities:
            data_train[m] = np.concatenate(data_train[m])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))

        return data_train, data_test, data_public

# ----- split ------
# 返回服务端数据和客户端数据
def split_server_train(data_train, dataset, ratio=0.1):
    y_train = data_train["y"]
    server_train = {m: np.empty((0, data_train[m].shape[1]))
                    for m in data_train if m != "y"}
    server_train["y"] = np.empty((0,))

    client_train = {m: np.empty((0, data_train[m].shape[1]))
                    for m in data_train if m != "y"}
    client_train["y"] = np.empty((0,))

    if dataset == "opp":
        n_div = N_LABEL_DIV_OPP
    elif dataset == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH
    elif dataset == "ur_fall":
        n_div = N_LABEL_DIV_URFALL
    elif dataset == "hapt":
        n_div = N_LABEL_DIV_HAPT
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_server_train = round(n_div * ratio)
    n_row = len(y_train)
    n_sample_per_div = n_row // n_div
    idxs = np.arange(0, n_row, n_sample_per_div)

    slices = {m: np.split(data_train[m], idxs) for m in data_train if m != "y"}
    slices_y = np.split(y_train, idxs)

    for m in slices:
        del slices[m][0]
    del slices_y[0]

    n_slices = len(slices_y)
    idxs_server_train = np.random.choice(np.arange(n_slices), n_server_train, replace=False)

    for i in range(n_slices):
        target = server_train if i in idxs_server_train else client_train
        for m in slices:
            target[m] = np.concatenate((target[m], slices[m][i]))
        target["y"] = np.concatenate((target["y"], slices_y[i]))

    return server_train, client_train


# split data with CLIENTS DATA (NO SERVER)
def split_clients_train(data_train, num_clients, dataset="default", overlap_ratio=0.3):
    """
    划分客户端数据集
    - data_train: dict, 每个模态和标签
    - num_clients: 客户端数量
    - dataset: 数据集名称，如果是 'ur_fall' 则允许重叠
    - overlap_ratio: 重叠比例（ur_fall专用）
    """
    y_train = data_train["y"]
    n_row = len(y_train)
    client_train_list = []

    if dataset == "ur_fall":
        # ur_fall 允许客户端之间有重叠
        seg_len = n_row // num_clients
        overlap_len = int(seg_len * overlap_ratio)

        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + seg_len
            # 防止越界
            if end_idx > n_row:
                end_idx = n_row
            client_data = {m: data_train[m][start_idx:end_idx] for m in data_train if m != "y"}
            client_data["y"] = y_train[start_idx:end_idx]
            client_train_list.append(client_data)

            # 下一客户端起始点前移一部分，实现重叠
            start_idx = start_idx + seg_len - overlap_len
            if start_idx >= n_row:
                start_idx = n_row - seg_len  # 最后一个客户端保证长度
    else:
        # 默认无重叠
        seg_len = n_row // num_clients
        for i in range(num_clients):
            start = i * seg_len
            end = (i + 1) * seg_len if i < num_clients - 1 else n_row
            client_data = {m: data_train[m][start:end] for m in data_train if m != "y"}
            client_data["y"] = y_train[start:end]
            client_train_list.append(client_data)

    return client_train_list


# 拿多少公共数据合适
def split_public(data_public, dataset="opp", ratio=1.):
    public_ratio = ratio
    y_public = data_public["y"]
    public_data = {m: np.empty((0, data_public[m].shape[1])) for m in data_public if m != "y"}
    public_data["y"] = np.empty((0))

    if dataset == "opp":
        n_div = N_LABEL_DIV_OPP
    elif dataset == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH
    elif dataset == "ur_fall":
        n_div = N_LABEL_DIV_URFALL
    elif dataset == "hapt":
        n_div = N_LABEL_DIV_HAPT
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    n_public = round(n_div * public_ratio)
    n_row = len(y_public)
    n_sample_per_div = n_row // n_div
    idxs = np.arange(0, n_row, n_sample_per_div)
    slices = {m: np.split(data_public[m], idxs) for m in data_public if m != "y"}
    slices_y = np.split(y_public, idxs)
    for m in slices:
        del slices[m][0]
    del slices_y[0]
    n_slices = len(slices_y)
    idxs_public = np.random.choice(np.arange(n_slices), n_public, replace=False)
    for i in range(n_slices):
        if i in idxs_public:
            for m in slices:
                public_data[m] = np.concatenate((public_data[m], slices[m][i]))
            public_data["y"] = np.concatenate((public_data["y"], slices_y[i]))
    return public_data


# 从data里随机抽取一个batch的序列数据
def make_seq_batch2(data, batch_size, seq_len=100):
    samples_y = data["y"]
    modalities = [m for m in data if m != "y"]
    input_sizes = {m: data[m].shape[1] for m in modalities}

    n_samples = len(samples_y)

    max_start = n_samples - seq_len
    if max_start <= 0:
        seq_len = min(50, n_samples)  # 防止 seq_len > 样本数
        max_start = n_samples - seq_len
    if max_start <= 0:
        # 样本量太小，直接重复取样
        indices_start = np.random.choice(n_samples, batch_size, replace=True)
    else:
        indices_start = np.random.choice(max_start, batch_size, replace=False)


  
    indices_start = np.random.choice(max_start, batch_size, replace=False)
    modalities_seq = {
        m: np.zeros((batch_size, seq_len, input_sizes[m]), dtype=np.float32)
        for m in modalities
    }
    y_seq = np.zeros((batch_size, seq_len), dtype=np.uint8)
    for i, idx_start in enumerate(indices_start):
        idx_end = idx_start + seq_len
        for m in modalities:
            modalities_seq[m][i, :, :] = data[m][idx_start: idx_end, :]
        y_seq[i, :] = samples_y[idx_start: idx_end]
    return modalities_seq, y_seq


def make_seq_batch(dataset, seg_idxs, seg_len, batch_size):
    """
    Returns:
        A tuple (modalities_seq, y_seq)
        - modalities_seq: dict, key 为模态名，value 形状为 (batch_size, seq_len, input_size)
        - y_seq: np.array, 形状为 (batch_size, seq_len)
    """
    samples_y = dataset["y"]
    modalities = [m for m in dataset if m != "y"]
    input_sizes = {m: dataset[m].shape[1] for m in modalities}

    # 计算序列长度
    seq_len_batch = seg_len * len(seg_idxs) // batch_size
    if seq_len_batch > seg_len:
        seq_len_batch = seg_len - 1

    # 收集起始索引
    all_indices_start = []
    for idx in seg_idxs:
        indices_start_in_seg = list(range(idx, idx + seg_len - seq_len_batch))
        all_indices_start.extend(indices_start_in_seg)
    indices_start = np.random.choice(all_indices_start, batch_size, replace=False)

    # 初始化 batch
    modalities_seq = {m: np.zeros((batch_size, seq_len_batch, input_sizes[m]), dtype=np.float32)
                      for m in modalities}
    y_seq = np.zeros((batch_size, seq_len_batch), dtype=np.uint8)

    # 生成 batch
    for i in range(batch_size):
        idx_start = indices_start[i]
        idx_end = idx_start + seq_len_batch
        for m in modalities:
            modalities_seq[m][i, :, :] = dataset[m][idx_start: idx_end, :]
        y_seq[i, :] = samples_y[idx_start: idx_end]

    return modalities_seq, y_seq


def client_idxs(data_train, dataset, num_clients):
    n_samples = len(data_train["y"])
    DATASET = dataset
    if DATASET == "opp":
        n_div = N_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_DIV_URFALL
    # each client has (n_samples * train_ratio) data
    train_ratio = TRAIN_RATIO

    len_div = int(n_samples // n_div) 
    len_seg = get_seg_len(n_samples, dataset)
    starts_div = np.arange(0, n_samples-len_div, len_div)
    idxs_clients = []
    for i in range(num_clients):
        idxs_clients.append(np.array([]).astype(np.int64))
        for start in starts_div:
            idxs_in_div = np.arange(start, start + len_div - len_seg)
            idxs_clients[i] = np.append(
                idxs_clients[i], np.random.choice(idxs_in_div))
    return idxs_clients


def get_seg_len(n_samples, dataset):
    DATASET = dataset
    train_ratio = TRAIN_RATIO
    if DATASET == "opp":
        n_div = N_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_DIV_URFALL
    return int(n_samples * float(train_ratio)//n_div)
# ==========================================================


def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)


def batch_data(data, batch_size):

    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if (len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if (sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset):

    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    test_data_dir = os.path.join('data', dataset, 'data', 'test')

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


def read_user_data(index, data, dataset):

    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]

    # X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if (dataset == "Epic"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(len(X_train), 1024).type(torch.float32)  # use for image
        # X_train = torch.Tensor(X_train).view(len(X_train), 400).type(torch.float32)  # use for amazon
        # X_train = torch.Tensor(X_train).view(len(X_train), 3, 32, 32).type(torch.float32) # use for Digit-five
        y_train = np.array(y_train, dtype=int)
        # X_test = torch.Tensor(X_test).reshape(len(X_test), 3, 32, 32).type(torch.float32)  # use for Digit-five
        X_test = torch.Tensor(X_test).reshape(len(X_test), 1024).type(torch.float32)  # use for image
        # X_test = torch.Tensor(X_test).reshape(len(X_test), 400).type(torch.float32)  # use for amazon
        y_test = np.array(y_test, dtype=int)
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        y_train = y_train.reshape(-1)   # use for office               #  remember change
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        y_test = y_test.reshape(-1)  # use for office

    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data


def print_dataset_info(dataset, name="Dataset"):
    """
    通用数据集信息打印函数。
    参数：
        dataset: dict，包含各模态数据和标签，如 {"A": array, "B": array, "y": array}
        name: str，可选，数据集名称（如 "Client 0"）
    """
    print(f"\n=== {name} ===")

    # 检查标签
    if "y" not in dataset:
        print("Warning: dataset does not contain key 'y'.")
        return

    # 各模态信息
    modalities = [k for k in dataset.keys() if k != "y"]
    print(f"Modalities: {modalities}")
    print(f"Total samples: {len(dataset['y'])}")

    # 各模态形状
    for m in modalities:
        print(f"  └─ {m}: shape {dataset[m].shape}")

    # 标签分布统计
    y_all = dataset["y"]
    unique_labels, counts = np.unique(y_all, return_counts=True)
    print("\nLabel distribution:")
    for l, c in zip(unique_labels, counts):
        print(f"  Class {l}: {c} samples ({c / len(y_all) * 100:.2f}%)")

    print("=" * 50)
