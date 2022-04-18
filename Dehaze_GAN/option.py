import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--root", default="./Dataset", help="NYU_N2 root")
    parser.add_argument("--train_logs", default="./Dehaze_GAN/Result/logs.txt",
                        help="Logs")
    parser.add_argument(
        "--save_root", default="./Dehaze_GAN/Result", help="save_root")

    parser.add_argument("--train_batch_size", default=1,
                        help="train batch size")
    parser.add_argument("--test_batch_size", default=4,
                        help="test batch size")

    parser.add_argument("--epochs", default=200, type=int, help="train epochs")
    parser.add_argument("--lr", default=1e-3, help="learning rate")
    parser.add_argument("--k", default=3, help="")
    parser.add_argument("--device", default="cuda", help="the train device")
    parser.add_argument("--momentum", default=0.9, help="momentum")
    parser.add_argument("--wd", default=1e-4, help="weight decay")
    parser.add_argument("--start_epoch", default=0, help="start epoch")
    parser.add_argument("--size", default=(128, 128), help="Crop size")
    parser.add_argument("--beta", default=(0.5, 0.9), help="optim beta")

    parser.add_argument("--haze_mean", default=(0.6980,
                        0.6817, 0.6766), help="the haze mean")
    parser.add_argument("--haze_std", default=(0.1906,
                        0.1958, 0.1974), help="the haze std")
    parser.add_argument("--img_mean", default=(0.4932,
                        0.4243, 0.4022), help="the img mean")
    parser.add_argument("--img_std", default=(0.1662,
                        0.1567, 0.1458), help="the img std")

    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--pos_every', action='store_true')
    parser.add_argument('--no_pos', action='store_true')
    parser.add_argument('--num_queries', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.post_norm)
    print(args.no_mlp)
    print(args.pos_every)
    print(args.no_pos)
