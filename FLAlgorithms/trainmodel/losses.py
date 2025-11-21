import torch
from torch.nn import functional as F

def contrastive_modality_align(z_proj_dict):
    # z_proj_dict[m] = [B, T, D]
    modalities = list(z_proj_dict.keys())
    align_loss = 0.0
    num_pairs = 0
    for i in range(len(modalities)):
        for j in range(i+1, len(modalities)):
            z1 = F.normalize(z_proj_dict[modalities[i]], dim=-1)  # [B, T, D]
            z2 = F.normalize(z_proj_dict[modalities[j]], dim=-1)
            # 对齐 loss = 1 - cosine similarity
            cos_sim = (z1 * z2).sum(dim=-1)  # [B, T]
            align_loss += (1 - cos_sim).mean()
            num_pairs += 1
    return align_loss / num_pairs

def compute_mmd(x, y, sigma_list=[1, 2, 4]):
    """
    x: [B1, D]
    y: [B2, D]
    MMD with multi-scale RBF kernels.
    """
    xx = torch.matmul(x, x.t())   # [B1, B1]
    yy = torch.matmul(y, y.t())   # [B2, B2]
    xy = torch.matmul(x, y.t())   # [B1, B2]

    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    Kxx, Kyy, Kxy = 0, 0, 0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)

        Kxx += torch.exp(-gamma * (rx.t() + rx - 2 * xx)).mean()
        Kyy += torch.exp(-gamma * (ry.t() + ry - 2 * yy)).mean()
        Kxy += torch.exp(-gamma * (rx.t() + ry - 2 * xy)).mean()

    return Kxx + Kyy - 2 * Kxy

def hsic_loss(z_share, z_spec):
    """
    计算批次级 HSIC 损失，鼓励 z_share 与 z_spec 统计独立
    z_share: [B, D]
    z_spec:  [B, D]
    """
    B = z_share.size(0)
    
    # 1. 线性核
    K = z_share @ z_share.t()  # [B, B]
    L = z_spec @ z_spec.t()    # [B, B]
    
    # 2. 中心化矩阵 H
    H = torch.eye(B, device=z_share.device) - (1.0 / B) * torch.ones(B, B, device=z_share.device)
    
    # 3. HSIC
    hsic = torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)
    
    return hsic



