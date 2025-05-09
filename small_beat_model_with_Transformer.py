import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


#  Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, dilation_rate, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, num_filters, kernel_size,
            padding=dilation_rate * (kernel_size - 1) // 2,
            dilation=dilation_rate,
        )
        self.conv2 = nn.Conv1d(
            num_filters, num_filters, kernel_size,
            padding=dilation_rate * (kernel_size - 1) // 2,
            dilation=dilation_rate,
        )
        self.conv_residual = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()

    def forward(self, x):
        res_x = self.conv_residual(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return res_x + x


# TCN
class TCN(nn.Module):
    def __init__(self, num_filters, kernel_size, dilations, dropout_rate=0.15):
        super().__init__()
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters, num_filters, kernel_size, d, dropout_rate) for d in dilations]
        )
        self.activation = nn.ELU()

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        return self.activation(x)


#BeatThisSmall with Transformer
class BeatThisSmall(nn.Module):
    def __init__(self, num_filters=20, kernel_size=5, num_dilations=10, dropout_rate=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, (3, 3), padding=(1, 0))
        self.pool1 = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (1, 20), padding=0)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.conv3 = nn.Conv2d(num_filters, num_filters, (3, 3), padding=(1, 0))
        self.pool3 = nn.MaxPool2d((1, 3))

        self.dropout = nn.Dropout(dropout_rate)

        # TCN
        dilations = [2 ** i for i in range(num_dilations)]
        self.tcn = TCN(num_filters, kernel_size, dilations, dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1500)

        # Transformer layer with dropout and internal LayerNorm
        self.positional_encoding = PositionalEncoding(num_filters)
        transformer_layer = TransformerEncoderLayer(
            d_model=num_filters,
            nhead=2,
            dim_feedforward=128,
            dropout=dropout_rate,      
            batch_first=True,
            norm_first=True            
        )
        self.transformer = TransformerEncoder(transformer_layer, num_layers=1)
        self.final_dropout = nn.Dropout(dropout_rate)  

        self.beats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)
        self.downbeats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, T, F]

        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.pool3(F.elu(self.conv3(x)))

        x = x.squeeze(-1)               # [B, C, T]
        x = self.tcn(x)                 # [B, C, T]
        x = self.adaptive_pool(x)       # [B, C, 1500]

        x = x.transpose(1, 2)           # [B, 1500, C]
        x = self.positional_encoding(x)
        x = self.transformer(x)         # [B, 1500, C]
        x = self.final_dropout(x)
        x = x.transpose(1, 2)           # [B, C, 1500]

        beats = self.beats_dense(x).squeeze(1)
        downbeats = self.downbeats_dense(x).squeeze(1)
        return {"beat": beats, "downbeat": downbeats}
