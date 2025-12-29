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

def hsic_loss(z_share, z_spec,sigma=None):
    # """
    # 计算批次级 HSIC 损失，鼓励 z_share 与 z_spec 统计独立
    # z_share: [B, D]
    # z_spec:  [B, D]
    # """
    # B = z_share.size(0)
    
    # # 1. 线性核
    # K = z_share @ z_share.t()  # [B, B]
    # L = z_spec @ z_spec.t()    # [B, B]
    
    # # 2. 中心化矩阵 H
    # H = torch.eye(B, device=z_share.device) - (1.0 / B) * torch.ones(B, B, device=z_share.device)
    
    # # 3. HSIC
    # hsic = torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)
    
    # return hsic
    
    import torch
    """
    基于高斯核 (RBF Kernel) 的 HSIC 损失
    z_share: [B, D]
    z_spec:  [B, D]
    sigma:   RBF 核的带宽，如果为 None 则使用中位数启发式 (Median Heuristic)
    """
    B = z_share.size(0)
    
    def compute_kernel(x, y):
        # 计算欧氏距离矩阵: ||x_i - y_j||^2
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist_sq = torch.clamp(dist_sq, min=0.0) # 防止数值误差产生负数
        
        # 中位数启发式自适应选择 sigma
        if sigma is None:
            # 取距离矩阵中位数作为带宽基准
            sigma_val = torch.median(dist_sq.detach())
            sigma_val = sigma_val if sigma_val > 1e-5 else 1.0 # 避免除零
        else:
            sigma_val = sigma ** 2
            
        # RBF 核: exp(-gamma * ||x-y||^2)
        gamma = 1.0 / (2.0 * sigma_val)
        kernel = torch.exp(-gamma * dist_sq)
        return kernel

    # 1. 计算核矩阵
    K = compute_kernel(z_share, z_share) # [B, B]
    L = compute_kernel(z_spec, z_spec)   # [B, B]
    
    # 2. 中心化矩阵 H
    H = torch.eye(B, device=z_share.device) - (1.0 / B) * torch.ones(B, B, device=z_share.device)
    
    # 3. 计算 HSIC: tr(KHLH)
    KH = torch.mm(K, H)
    LH = torch.mm(L, H)
    hsic = torch.trace(torch.mm(KH, LH)) / ((B - 1) ** 2)
    
    return hsic



