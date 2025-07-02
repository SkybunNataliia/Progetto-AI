import torch.nn as nn
import torch

class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_filters=[32, 64], kernel_size=3, stride=1, padding=1, dropout=0.3):
        super(CNN1DModel, self).__init__()
        layers = []
        in_channels = input_size

        for out_channels in num_filters:
            layers.append(nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters[-1] * 2, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        # Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # swap seq_len e features
        out = self.conv(x)      # [batch, channels, seq_len]
        
        # global average pooling e global max pooling lungo seq_len
        avg_pool = out.mean(dim=2)
        max_pool, _ = out.max(dim=2)
        out = torch.cat([avg_pool, max_pool], dim=1)  # concatena feature
        
        out = self.fc(out)      # [batch, 1]
        return out.squeeze(1)