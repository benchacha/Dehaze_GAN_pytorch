import torch
from train import train
from option import parse_args
from NYU_v2 import Trans, NYU_v2
from torch.utils.data import DataLoader
from models.Dehaze_gan import Generator, Discriminator


def main():
    args = parse_args()
    args.epochs = 50

    genernator = Generator()
    discriminator = Discriminator()

    device = torch.device(args.device)
    genernator.to(device)
    discriminator.to(device)

    transform = Trans(size=args.size)

    train_dataset = NYU_v2(args.root, set="train", transform=transform)
    test_dataset = NYU_v2(args.root, set="test", transform=transform)

    train_loader = DataLoader(
        train_dataset, args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.test_batch_size)

    gen_optim = torch.optim.Adam(
        genernator.parameters(), args.lr, betas=args.beta)
    dis_optim = torch.optim.Adam(
        discriminator.parameters(), args.lr, betas=args.beta)

    train(genernator, discriminator, gen_optim,
          dis_optim, train_loader, test_loader, args.k, args.epochs, device, args.save_root)


if __name__ == "__main__":
    main()
