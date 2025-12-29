import os
import torch
import torch.nn.functional as F
import multiprocessing

from torch import nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

Num_neurons = 32

class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
    
    
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, representation_size, num_layers=1, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=Num_neurons,
                            num_layers=num_layers, batch_first=batch_first)
        self.lstm2 = nn.LSTM(input_size=Num_neurons, hidden_size=representation_size,
                             num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out_im, _ = self.lstm(x)
        out, _ = self.lstm2(out_im)
        return out_im, out  # 先输出中间层，再输出最终层


class LSTMDecoder(nn.Module):
    def __init__(self, representation_size, output_size, num_layers=1, batch_first=True):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=representation_size, hidden_size=Num_neurons,
                            num_layers=num_layers, batch_first=batch_first)
        self.lstm2 = nn.LSTM(input_size=Num_neurons, hidden_size=output_size,
                             num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out1, _ = self.lstm2(out)
        return out1

# IN USE


class SplitLSTMAutoEncoder(nn.Module):
    def __init__(self, input_sizes, representation_size, num_layers=1, batch_first=True):
        """
        input_sizes: dict, key为模态名，value为特征维度
        representation_size: int, LSTM表征维度
        """
        super().__init__()
        self.batch_first = batch_first
        self.modalities = list(input_sizes.keys())

        # 动态创建每个模态的 encoder/decoder
        self.encoders = nn.ModuleDict({
            m: LSTMEncoder(input_size=input_sizes[m],
                           representation_size=representation_size,
                           num_layers=num_layers,
                           batch_first=batch_first)
            for m in self.modalities
        })
        self.decoders = nn.ModuleDict({
            m: LSTMDecoder(representation_size=representation_size,
                           output_size=input_sizes[m],
                           num_layers=num_layers,
                           batch_first=batch_first)
            for m in self.modalities
        })

    def info(self):
        print("AutoEncoder with modalities:", self.modalities)
        for m in self.modalities:
            print(f" Modality: {m}, Encoder: {self.encoders[m]}, Decoder: {self.decoders[m]}")

    def forward(self, x, modality):
        assert modality in self.modalities, f"Modality {modality} not found"

        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        _, out = self.encoders[modality](x)

        # 取最后时间步作为表示
        representation = out[:, -1, :].unsqueeze(1) if self.batch_first else out[-1, :, :].unsqueeze(0)
        representation_seq = representation.expand(-1, seq_len, -1)
        # 返回该模态对应的 decoder 输出
        x_reconstructed = self.decoders[modality](representation_seq)
        return x_reconstructed

    def encode(self, x, modality):
        assert modality in self.modalities, f"Modality {modality} not found"
        out, out1 = self.encoders[modality](x)
        return out, out1  #MID , FINAL

    def decode(self, h, modality, seq_len=None):
        assert modality in self.modalities, f"Modality {modality} not found"
        if seq_len is None:
            seq_len = 1
        if h.dim() == 2:
            h = h.unsqueeze(1).expand(-1, seq_len, -1)
        elif h.dim() == 3 and h.shape[1] != seq_len:
            h = h.expand(-1, seq_len, -1)
        return self.decoders[modality](h)

class DisentangledLSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, representation_size, shared_size, specific_size, num_layers=1, batch_first=True):
        super(DisentangledLSTMAutoEncoder, self).__init__()

        self.encoder = LSTMEncoder(input_size=input_size,
                                   representation_size=representation_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first)
        
        # === [修改点 1] 解码器输入维度变了 ===
        # 以前只有 specific_size，现在是 shared_size + specific_size
        self.decoder = LSTMDecoder(representation_size=shared_size + specific_size, 
                                   output_size=input_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first)

        self.fc_share = nn.Linear(representation_size, shared_size)
        self.fc_spec = nn.Linear(representation_size, specific_size)

        nn.init.orthogonal_(self.fc_share.weight)
        nn.init.orthogonal_(self.fc_spec.weight)

    # encode 和 encode_recon 保持不变 ...
    def encode(self, x, modality=None):
        _, out = self.encoder(x)
        z_share = self.fc_share(out)
        z_spec = self.fc_spec(out)
        return z_share, z_spec, out

    def encode_recon(self, x, modality=None):
        _, out = self.encoder(x)
        h_last = out[:, -1, :]
        z_share = self.fc_share(h_last)
        z_spec = self.fc_spec(h_last)
        return z_share, z_spec, out

    # === [修改点 2] decode 函数接收两者并拼接 ===
    def decode(self, z_share, z_spec, seq_len):
        # 1. 对 z_share 进行梯度缩放 (0.1 表示只接受 10% 的重构梯度)
        # 这样 z_share 只学轮廓，不记噪声
        z_share_scaled = GradScaler.apply(z_share, 1.0)
        
        # 2. 拼接 shared 和 specific
        z_cat = torch.cat([z_share_scaled, z_spec], dim=-1)
        
        # 3. 扩展维度送入解码器
        z_seq = z_cat.unsqueeze(1).expand(-1, seq_len, -1)
        x_recon = self.decoder(z_seq)  # B S D
        return x_recon

    # === [修改点 3] forward 传入 z_share ===
    def forward(self, x, modality=None):
        z_share, z_spec, out = self.encode_recon(x)
        seq_len = x.shape[1]
        
        # 修改调用方式
        x_recon = self.decode(z_share, z_spec, seq_len)
        
        return x_recon, z_share, z_spec
    
# class DisentangledLSTMAutoEncoder(nn.Module):
#     def __init__(self, input_size, representation_size, shared_size, specific_size, num_layers=1, batch_first=True):
#         super(DisentangledLSTMAutoEncoder, self).__init__()

#         self.encoder = LSTMEncoder(input_size=input_size,
#                                    representation_size=representation_size,
#                                    num_layers=num_layers,
#                                    batch_first=batch_first)
#         self.decoder = LSTMDecoder(representation_size=specific_size,
#                                    output_size=input_size,
#                                    num_layers=num_layers,
#                                    batch_first=batch_first)

#         # 线性映射：从 encoder 输出得到共享特征和私有特征
#         self.fc_share = nn.Linear(representation_size, shared_size)
#         self.fc_spec = nn.Linear(representation_size, specific_size)

#         # 初始化为正交，鼓励两种特征独立
#         nn.init.orthogonal_(self.fc_share.weight)
#         nn.init.orthogonal_(self.fc_spec.weight)

#     def encode(self, x, modality=None):
#         _, out = self.encoder(x)
#         z_share = self.fc_share(out)
#         z_spec = self.fc_spec(out)
#         return z_share, z_spec, out

#     def encode_recon(self, x, modality=None):
#         _, out = self.encoder(x)
#         h_last = out[:, -1, :]  # 取最后时间步的隐藏状态
#         z_share = self.fc_share(h_last)
#         z_spec = self.fc_spec(h_last)
#         return z_share, z_spec, out
        
        
#     # def encode(self, x, modality=None):
#     #     # x.shape: (batch_size, seq_len, input_size)

#     #     _, out = self.encoder(x)
#     #     h_last = out[:, -1, :]  # 取最后时间步的隐藏状态
#     #     z_share = self.fc_share(h_last)
#     #     z_spec = self.fc_spec(h_last)
#     #     return z_share, z_spec, out

#     def decode(self, z_spec, seq_len):
#         z_seq = z_spec.unsqueeze(1).expand(-1, seq_len, -1)
#         x_recon = self.decoder(z_seq)  # B S D
#         return x_recon

#     def forward(self, x, modality=None):
#         z_share, z_spec, out = self.encode_recon(x)
#         seq_len = x.shape[1]
#         x_recon = self.decode(z_spec, seq_len)
#         return x_recon, z_share, z_spec


# ================================================================================================

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, n_classes)

    def forward(self, x):
        # print(x.shape,self.input_size,self.n_classes)
        x = self.dropout(x)
        out = self.fc(x)
        out = out.contiguous().view(-1, self.n_classes)
        return F.log_softmax(out, dim=1)

class FusionNet(nn.Module):
    def __init__(self, modalities, rep_size):
        super().__init__()
        self.modalities = modalities
        # 每个模态一个可学习 scalar
        self.weights = nn.Parameter(torch.ones(len(modalities)))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, z_dict):
        """
        z_dict[m] = [B, D]
        """
        w = self.softmax(self.weights)        # [M]
        z_list = list(z_dict.values())        # [M * [B, D]]
        fused = sum(w[i] * z_list[i] for i in range(len(z_list)))
        return fused          # [B, D]

import torch
import torch.nn as nn

class DynamicGatedFusion(nn.Module):
    def __init__(self, modalities, rep_size, hidden_dim=64):
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.rep_size = rep_size
        
        # 1. 计算总输入维度：所有模态拼接后的长度
        total_input_dim = self.num_modalities * rep_size
        
        # 2. 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Softmax(dim=-1)  # 在最后一个维度(模态维度)归一化
        )

    def forward(self, z_dict):

        try:
            z_list = [z_dict[m] for m in self.modalities]
        except KeyError as e:
            raise KeyError(f"Missing modality key: {e}. Expected: {self.modalities}")

        cat_z = torch.cat(z_list, dim=-1)
        weights = self.gate_net(cat_z) 
        stack_z = torch.stack(z_list, dim=-2)
        weights_expanded = weights.unsqueeze(-1)
        z_fused = torch.sum(stack_z * weights_expanded, dim=-2)
        return z_fused


class StyleAwareGenerator(nn.Module):
    def __init__(self, style_dim, n_classes, hidden_dim, content_dim):
        """
        style_dim: z_spec 的维度 (通常等于 specific_size)
        content_dim: z_share 的维度 (通常等于 shared_size/rep_size)
        """
        super(StyleAwareGenerator, self).__init__()
        
        # 标签嵌入层: 将类别 label 映射为向量
        self.label_emb = nn.Embedding(n_classes, style_dim)
        
        # 生成网络: 输入 (Style噪声 + Label嵌入) -> 输出 (模拟的 z_share)
        self.net = nn.Sequential(
            nn.Linear(style_dim * 2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, content_dim) 
            # 输出直接对齐 z_share (通常是经过 mean-pooling 后的向量)
        )

    def forward(self, z_style, labels):
        # z_style: 从客户端分布采样得到的 [B, style_dim]
        # labels: [B]
        
        c = self.label_emb(labels) # [B, style_dim]
        
        # 拼接 风格噪声 和 标签信息
        x = torch.cat([z_style, c], dim=1) 
        
        out = self.net(x)
        return out