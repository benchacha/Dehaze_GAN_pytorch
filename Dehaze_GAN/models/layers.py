import torch
import torch.nn as nn
# 符合层


class ConvBlocks(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, drop_rate):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.Dropout2d(drop_rate))

    def forward(self, x):
        return self.conv(x)


# DB
class DenseBlock(nn.Module):

    def __init__(self, in_ch, num_conv=4, growth_gate=12,
                 drop_rate=0.2, efficient=False):
        super(DenseBlock, self).__init__()
        # 复合层的层数
        self.num_conv = num_conv
        # 增长率
        self.in_ch = in_ch
        self.growth_gate = growth_gate
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.layers = nn.Sequential(*[
            nn.Sequential(
                ConvBlocks(self.in_ch + i * self.growth_gate,
                           4 * self.growth_gate, 1, 1, 0, self.drop_rate),
                ConvBlocks(4 * self.growth_gate, self.growth_gate, 3, 1, 1,
                           self.drop_rate)) for i in range(self.num_conv)
        ])

    def forward(self, x):
        if self.efficient:
            new_feature = []
            for layer in self.layers:
                out = layer(x)
                x = torch.concat([x, out], dim=1)
                new_feature.append(out)
            return torch.concat(new_feature, dim=1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.concat([x, out], dim=1)
            return x


# TD
class TransitionDown(nn.Module):

    def __init__(self, in_ch, kernel_size=1, stride=1, padding=0, drop_rate=0.2):
        super(TransitionDown, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding),
            nn.Dropout2d(drop_rate), nn.AvgPool2d(2))

    def forward(self, x):
        return self.layer(x)


# TU
class TransitionUp(nn.Module):

    def __init__(self, in_ch, kernel_size=4, stride=2, padding=1):
        super(TransitionUp, self).__init__()
        self.in_ch = in_ch
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride, padding))

    def forward(self, x):
        return self.layer(x)


class EncoderLayer(nn.Module):

    def __init__(self, in_ch, num_conv, growth_rate, drop_rate):
        super(EncoderLayer, self).__init__()

        self.TD = TransitionDown(in_ch, drop_rate=drop_rate)
        self.DB = DenseBlock(in_ch, num_conv, growth_rate, drop_rate)

    def forward(self, x):
        return self.DB(self.TD(x))


class DecoderLayer(nn.Module):

    def __init__(self, in_ch, num_conv, growth_rate, drop_rate):
        super(DecoderLayer, self).__init__()
        self.DB = DenseBlock(in_ch, num_conv, growth_rate,
                             drop_rate, efficient=True)
        self.TU = TransitionUp(num_conv * growth_rate)

    def forward(self, x):

        return self.TU(self.DB(x))
