import os
import string
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tools.visualization import handle_img, unormalize


class NYU_v2(Dataset):

    def __init__(self, root: string, set: string, transform=None):
        super(NYU_v2, self).__init__()
        # 集合
        self.set = set
        self.transform = transform
        filepath = os.path.join(root, 'NYU_v2_dataset', set)
        assert os.path.exists(filepath) == True, "{} can't find.".format(
            filepath)

        names = [
            name.split('_')[0]
            for name in os.listdir(os.path.join(filepath, 'Image'))
        ]

        self.haze_imgs = [
            os.path.join(filepath, 'Haze_Image', name + '_' + set + '.jpg')
            for name in names
        ]

        self.imgs = [
            os.path.join(filepath, 'Image', name + '_' + set + '.jpg')
            for name in names
        ]

        assert len(self.haze_imgs) == len(self.imgs), '{} , {}'.format(
            len(self.haze_imgs), len(self.imgs))

        for i in range(len(self.imgs)):
            assert self.haze_imgs[i].split('_')[0] == self.imgs[i].split(
                '_')[0], '{}:{}'.format(self.haze_imgs[i], self.imgs[i])

    def __getitem__(self, index):

        haze_img = Image.open(self.haze_imgs[index]).convert('RGB')
        img = Image.open(self.imgs[index]).convert('RGB')

        if self.transform is not None:
            haze_img, img = self.transform(haze_img, img)

        return haze_img, img

    def __len__(self):
        return len(self.imgs)


class Trans():

    def __init__(self, size=(256, 256)):

        self.haze_trans = transforms.Compose([
            transforms.CenterCrop((440, 600)),
            transforms.Resize(size=size),
            transforms.ToTensor(),
        ])

        self.img_trans = transforms.Compose([
            transforms.CenterCrop((440, 600)),
            transforms.Resize(size=size),
            transforms.ToTensor(),
        ])

    def __call__(self, haze_img, img):

        haze_img = self.haze_trans(haze_img) * 2 - 1
        img = self.img_trans(img) * 2 - 1
        return haze_img, img
