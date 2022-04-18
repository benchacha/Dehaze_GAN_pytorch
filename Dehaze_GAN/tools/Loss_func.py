import torch
import torchvision
import torch.nn as nn


class Vgg19(nn.Module):

    def __init__(self):
        super(Vgg19, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.layers = nn.Sequential(*list(model.children())[:1])

    def forward(self, x):
        output = self.layers(x)
        return output

# gen


class Gen_Loss(nn.Module):

    def __init__(self, weight=None, device=None):
        super(Gen_Loss, self).__init__()
        if device == None:
            device = torch.device("cpu")
        self.weight = weight
        self.L1_loss = nn.L1Loss().to(device)
        self.Vgg_loss = nn.MSELoss(reduction='mean').to(device)
        self.eps = 10e-12
        self.C = 1e-5
        self.Vgg19 = Vgg19().to(device)

    def forward(self, img, gen_img, dis_feak_result):
        lg = -torch.mean(torch.log(dis_feak_result + self.eps))
        l1 = self.L1_loss(img, gen_img)
        vgg_loss = self.C * self.Vgg_loss(
            self.Vgg19(gen_img).detach(),
            self.Vgg19(img).detach())

        return lg * self.weight[0] + l1 * self.weight[
            1] + vgg_loss * self.weight[2]


if __name__ == '__main__':
    model = Vgg19()
    input = torch.randn(1, 3, 256, 256)
    output = model(input)

    print(output.shape)
