import argparse

from .training import train


def main():

    parser = argparse.ArgumentParser(
        prog='Project training',
        description='This program manages training for my BSc Thesis project.',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True, type=str)
    parser.add_argument('-i', '--interval', type=int)

    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', required=True)

    parser.add_argument('--lr', type=float)
    parser.add_argument('-b', '--batch')
    parser.add_argument('-e', '--epochs', required=True, type=int)

    parser.add_argument('--memory')

    args = parser.parse_args()

    kwargs = {
        "model": args.model,
        "epochs": args.epochs,
        "device": args.device,
        "checkpoint_path": args.checkpoint,
    }
    if args.lr: kwargs["lr"] = args.lr
    if args.batch: kwargs["batch_size"] = args.batch
    if args.memory: kwargs["memory_checkpoint_path"] = args.memory
    if args.interval: kwargs["checkpoint"] = args.interval

    train(**kwargs)

if __name__ == "__main__":
    main()
