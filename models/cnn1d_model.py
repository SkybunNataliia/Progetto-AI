import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_channels=[32, 64], kernel_size=3, dropout=0.2):
        super(CNN1DModel, self).__init__()
        layers = []
        in_channels = input_size

        for out_channels in num_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        # Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # swap seq_len e features
        out = self.conv(x)      # [batch, channels, seq_len]
        out = out.mean(dim=2)   # global average pooling su seq_len
        out = self.fc(out)      # [batch, 1]
        return out.squeeze(1)