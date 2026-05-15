import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from project.models import EDMEvelynn
from project.train.config import DiffTrainingConfig
from project.util.data import ReplayMemoryData
from project.util.weights_and_biases import WandB


def train_diffusion(
        checkpoint_path: str,
        memory_checkpoint_path: str,
        device: str,
        epochs: int,
        evaluation_dir: str,
        tag: str = "",
) -> None:

    config = DiffTrainingConfig()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
	transforms.Pad(2),
    ])

    print("Loading data...")
    data = ReplayMemoryData(
        memory=memory_checkpoint_path,
        transform=transform,
        cap=config.cap
    )
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    print(f"Loaded data!")

    print("Initializing model...")
    model = EDMEvelynn(
        img_resolution=config.img_resolution,
        img_channels=config.img_channels,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        num_blocks=config.num_blocks,
        attn_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        batch_size=config.batch_size,
        lr=config.lr,
        network=config.network,
        mixed_precision=config.mixed_precision
    ).to(device)

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        model.load(checkpoint_path)
        print("Loaded checkpoint!")
    print("Initialized model!")

    wandb = WandB(
        project="Diff",
        lr=config.lr,
        network=config.network,
        in_channels=config.img_channels,
        start_channels=config.model_channels,
        channel_multipliers=str(config.channel_mult),
        attention_resolutions=str(config.attn_resolutions),
        num_res_blocks=config.num_blocks,
        tag=tag,
    )

    model.train(epochs, loader, wandb=wandb, evaluation_dir=evaluation_dir)

    wandb.finish()

    print("Saving checkpoint...")
    model.save(checkpoint_path)
    print("Checkpoint saved!")


def main():

    parser = argparse.ArgumentParser(
        prog='Project train (Diffusion)',
        description='This program manages train for my BSc Thesis project (Diffusion).',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True, type=str)

    parser.add_argument('-d', '--device', required=True, type=str)

    parser.add_argument('-e', '--epochs', required=True, type=int)

    parser.add_argument('-m', '--memory', required=True, type=str)

    parser.add_argument('-t', '--tag', default="", type=str)

    parser.add_argument('--dir', default=None, type=str)

    args = parser.parse_args()

    train_diffusion(
        checkpoint_path=args.checkpoint,
        memory_checkpoint_path=args.memory,
        device=args.device,
        epochs=args.epochs,
        evaluation_dir=args.dir,
        tag=args.tag,
    )

if __name__ == "__main__":
    main()
