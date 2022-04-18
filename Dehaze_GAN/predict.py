import torch
from zmq import device
from NYU_v2 import NYU_v2, Trans
from models.Dehaze_gan import Generator
from tools.visualization import vis_Results


if __name__ == "__main__":
    root = './Dataset'
    save_root = './Dehaze_GAN/Result'

    size = [256, 256]
    transform = Trans(size=size)

    test_dataset = NYU_v2(root, set='train', transform=transform)
    # 生成器
    generator = Generator()
    # 加载模型参数
    generator.load_state_dict(torch.load('./Dehaze_GAN/Result/gen_8.pth'))

    num, device = 4, torch.device('cuda')

    generator.to(device)

    vis_Results(generator, rows=3, dataset=test_dataset,
                num=num, device=device)
