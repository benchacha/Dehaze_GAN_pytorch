import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def handle_img(img):

    if len(img.shape) == 3:

        img = img.permute(1, 2, 0)
    img_np = img.detach().cpu().numpy()

    if len(img_np.shape) == 2:
        img_np[img_np == 255] = 0
        img_np[img_np == 1] = 255
    img_PIL = Image.fromarray(np.uint8(img_np))
    return img_PIL


def show_imgs(images, rows, cols):
    plt.figure(figsize=(8, 5))
    for row in range(rows):
        for col in range(cols):
            plt.subplot(rows, cols, row * cols + col + 1)

            if row != 0:

                plt.imshow(images[col][row], cmap='gray')
            else:
                plt.imshow(images[col][row])
            plt.xticks([])
            plt.yticks([])
    plt.show()


def unormalize(img, channel=3, mean=(0, 0, 0), std=(1, 1, 1), device=None):
    mean = torch.tensor(mean).reshape(channel, 1, 1).to(device)
    std = torch.tensor(std).reshape(channel, 1, 1).to(device)
    return (img * std + mean) * 255


def predict(model, img, device):

    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0)
        img = img.to(device)
        pred = model(img)
        return pred.squeeze(0)


def vis_Results(model, rows=3, dataset=None, num=1, device=None):
    imgs = []

    for i in range(num):

        haze_img, img = dataset[i]
        haze_img = haze_img.to(device)
        img = img.to(device)

        gen_img = predict(model, haze_img, device)
        img = (img + 1) / 2
        haze_img = (haze_img + 1) / 2
        gen_img = (gen_img + 1) / 2

        haze_img = unormalize(haze_img, 3, device)
        gen_img = unormalize(gen_img, 3, device)
        img = unormalize(img, 3, device)

        imgs.append(
            [handle_img(haze_img),
             handle_img(img),
             handle_img(gen_img)])
    show_imgs(imgs, rows, num)
