import torch
import torch.nn as nn
from models.layers import EncoderLayer, DecoderLayer
from models.layers import DenseBlock, TransitionDown, TransitionUp


class Generator(nn.Module):

    def __init__(self, in_ch=3, basic_ch=48, num_classes=3, num_conv=4, bot_num=15, growth_rate=12, drop_rate=0.2):
        super(Generator, self).__init__()

        grow_conv = num_conv * growth_rate

        basic_ch = basic_ch

        self.conv1 = nn.Conv2d(in_ch, basic_ch, 3, 1, 1)

        self.DB1 = DenseBlock(basic_ch, num_conv, growth_rate, drop_rate)

        self.encoder = nn.ModuleList([
            EncoderLayer(in_ch=basic_ch + grow_conv * (i + 1), num_conv=num_conv,
                         growth_rate=growth_rate, drop_rate=drop_rate)
            for i in range(4)]
        )

        self.bottleneck = nn.Sequential(
            TransitionDown(basic_ch + grow_conv * 5, drop_rate=drop_rate),
            DenseBlock(basic_ch + grow_conv * 5, bot_num,
                       growth_rate, drop_rate, efficient=True),
            TransitionUp(bot_num * growth_rate)
        )

        self.decoder = nn.ModuleList([
            DecoderLayer(in_ch=basic_ch + grow_conv *
                         (6 - i) if i else basic_ch +
                         grow_conv * 5 + bot_num * growth_rate,
                         num_conv=num_conv, growth_rate=growth_rate, drop_rate=drop_rate)
            for i in range(4)
        ])

        self.DB10 = DenseBlock(basic_ch + grow_conv * 2, num_conv,
                               growth_rate, drop_rate, efficient=False)

        self.final_conv = nn.Conv2d(
            basic_ch + grow_conv * 3, num_classes, 1, 1, 0)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x_skip = []
        x = self.DB1(self.conv1(x))
        x_skip.append(x)

        for layer in self.encoder:
            x = layer(x)
            x_skip.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            x = layer(torch.concat([x_skip[4 - i], x], dim=1))

        output = self.final_conv(
            self.DB10(torch.concat([x_skip[0], x], dim=1)))

        return torch.tanh(output)


# Batchnorm, leaky relu, conv
class BLC(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super(BLC, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))

    def forward(self, x):

        return self.layer(x)


class Discriminator(nn.Module):

    def __init__(self, basic_ch=64):
        super(Discriminator, self).__init__()
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(6, basic_ch, 1, 1, 0)
        self.layer1 = BLC(basic_ch, basic_ch * 2, 3, 2, 1)
        self.layer2 = BLC(basic_ch * 2, basic_ch * 4, 3, 2, 1)
        self.layer3 = BLC(basic_ch * 4, basic_ch * 8, 3, 2, 1)
        self.conv2 = nn.Conv2d(basic_ch * 8, 1, 3, 1, 0)
        # 初始化参数
        self.apply(_init_vit_weights)

    def forward(self, haze_img, img):
        input = torch.concat([img, haze_img], dim=1)
        x = self.conv1(self.leaky_relu1(input))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.conv2(x)

        return torch.sigmoid(x)


def _init_vit_weights(m):
    """
    model weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == '__main__':
    generator = Generator()

    discriminator = Discriminator()

    haze_img = torch.rand(4, 3, 256, 256)

    output = generator(haze_img)
    output1 = discriminator(output, haze_img)
    print('gen_output:'.ljust(20), output.shape)
    print('dis_output:'.ljust(20), output1.shape)
