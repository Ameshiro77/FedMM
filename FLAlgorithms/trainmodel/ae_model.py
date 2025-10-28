import os
import torch
import torch.nn.functional as F
import multiprocessing

from torch import nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

Num_neurons = 32


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
        return out, out1

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
        self.decoder = LSTMDecoder(representation_size=specific_size,
                                   output_size=input_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first)

        # 线性映射：从 encoder 输出得到共享特征和私有特征
        self.fc_share = nn.Linear(representation_size, shared_size)
        self.fc_spec = nn.Linear(representation_size, specific_size)

        # 初始化为正交，鼓励两种特征独立
        nn.init.orthogonal_(self.fc_share.weight)
        nn.init.orthogonal_(self.fc_spec.weight)

    def encode(self, x, modality=None):
        # x.shape: (batch_size, seq_len, input_size)
        
        _, out = self.encoder(x)
        h_last = out[:, -1, :]  # 取最后时间步的隐藏状态
        z_share = self.fc_share(h_last)
        z_spec = self.fc_spec(h_last)
        return z_share, z_spec, out

    def decode(self, z_spec, seq_len):
        z_seq = z_spec.unsqueeze(1).expand(-1, seq_len, -1)
        x_recon = self.decoder(z_seq)  # B S D
        return x_recon

    def forward(self, x, modality=None):
        z_share, z_spec, out = self.encode(x)
        seq_len = x.shape[1]
        x_recon = self.decode(z_spec, seq_len)
        return x_recon, z_share, z_spec


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


