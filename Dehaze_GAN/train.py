import os
import copy
import time
import torch
from piq import ssim, psnr
from tools.logs import write_infor
from tools.Loss_func import Gen_Loss


# train discriminator
def train_dis(generator, discriminator, dis_optim,
              haze_img, img, device=None):

    eps = 10e-12

    discriminator.train()
    dis_optim.zero_grad()
    real_result = discriminator(haze_img, img)
    gen_img = generator(haze_img).detach()
    feak_result = discriminator(haze_img, gen_img)

    loss_D = -torch.mean(
        torch.log(real_result + eps) + torch.log(1 - feak_result + eps))
    loss_D = loss_D.to(device)
    loss_D.backward(retain_graph=True)
    dis_optim.step()
    return loss_D.item()


# train generator
def train_gen(generator, discriminator, gen_optim, loss_func,
              haze_img, img, device=None):
    generator.train()
    gen_optim.zero_grad()

    gen_img = generator(haze_img)
    feak_result = discriminator(haze_img, gen_img)
    loss_G = loss_func(img, haze_img, feak_result)
    loss_G.backward(retain_graph=True)
    gen_optim.step()

    return loss_G.item()


# train one epoch
def train_epoch(generator, discriminator, gen_optim,
                dis_optim, train_loader, k=5, device=None, save_root=None):

    file_name = "result.txt"
    weight = [2, 100, 10]

    loss_func = Gen_Loss(weight, device)

    for i, (haze_img, img) in enumerate(train_loader):

        haze_img = haze_img.to(device)
        img = img.to(device)

        loss_D = train_dis(generator, discriminator, dis_optim, haze_img,
                           img, device)
        loss_T = train_gen(generator, discriminator, gen_optim, loss_func,
                           haze_img, img, device)
        if i % 400 == 0:
            write_infor(save_root, file_name,
                        'batch_{}: loss_D'.format(i).ljust(10), loss_D)
            write_infor(save_root, file_name,
                        'batch_{}: loss_T'.format(i).ljust(10), loss_T)


def test_epoch(generator, test_loader, device=None, save_root=None):
    file_name = "result.txt"
    generator.eval()
    with torch.no_grad():
        length = len(test_loader)
        psnr_sum = 0.
        ssim_sum = 0.

        for haze_imgs, imgs in test_loader:
            haze_imgs = haze_imgs.to(device)
            imgs = imgs.to(device)
            gen_imgs = generator(haze_imgs)

            imgs = (imgs + 1) / 2 * 255
            gen_imgs = (gen_imgs + 1) / 2 * 255

            psnr_sum += psnr(imgs, gen_imgs, data_range=255.)
            ssim_sum += ssim(imgs, gen_imgs, data_range=255.)

    psnr_mean = psnr_sum.item() / length
    ssim_mean = ssim_sum.item() / length
    score = 0.05 * psnr_mean + ssim_mean
    write_infor(save_root, file_name, 'psnr', psnr_mean)
    write_infor(save_root, file_name, 'ssim', ssim_mean)
    write_infor(save_root, file_name, 'test score', score)
    return score


def train(generator, discriminator, gen_optim, dis_optim,
          train_loader, test_loader, k=1, epochs=3, device=None,
          save_root=None):
    print('train on  {} .'.format(device))
    filename = 'result.txt'

    best_score = score = 0.
    best_epoch = 0
    best_gen_weight = copy.deepcopy(generator.state_dict())
    best_dis_weight = copy.deepcopy(discriminator.state_dict())

    start_time = time.time()

    write_infor(save_root, filename, 'start_time', start_time)
    for epoch in range(epochs):
        print('\nthe {}th epoch'.format(epoch))
        write_infor(save_root, filename, '\nEpoch', epoch)

        train_epoch(generator, discriminator, gen_optim, dis_optim,
                    train_loader, k, device, save_root)

        score = test_epoch(generator, test_loader, device, save_root)

        if score > best_score:
            best_epoch = epoch
            best_score = score
            best_gen_weight = copy.deepcopy(generator.state_dict())
            best_dis_weight = copy.deepcopy(discriminator.state_dict())

        torch.save(best_gen_weight, os.path.join(save_root, 'gen.pth'))

        torch.save(best_dis_weight, os.path.join(save_root, 'dis.pth'))

        print('{} : end.'.format(epoch))

    end_time = time.time()
    write_infor(save_root, filename, 'end_time', end_time)
    write_infor(save_root, filename, 'duration', end_time - start_time)

    torch.save(best_gen_weight, os.path.join(
        save_root, 'gen_{}.pth'.format(best_epoch)))

    torch.save(best_dis_weight, os.path.join(
        save_root, 'dis_{}.pth'.format(best_epoch)))
