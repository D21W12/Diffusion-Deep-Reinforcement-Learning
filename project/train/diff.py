import argparse
import os
from time import sleep

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from project.models import EDMEvelynn
from project.train.config import DiffTrainingConfig
from project.util.data import ReplayMemoryData
from project.util.transforms import ToFloat
from project.util.weights_and_biases import WandB


def train_diffusion(
        checkpoint_path: str,
        memory_checkpoint_path: str,
        device: str,
        epochs: int,
        tag: str = "",
) -> None:

    config = DiffTrainingConfig()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    print("Loading data...")
    data = ReplayMemoryData(
        memory=memory_checkpoint_path,
        transform=transform,
        cap=100_000
    )
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    print(f"Loaded data!")

    print("Initializing model...")
    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attn_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        batch_size=config.batch_size,
        lr=config.lr,
        network=config.network
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
        in_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_multipliers=config.channel_multipliers,
        attention_resolutions=config.attention_resolutions,
        num_res_blocks=config.num_res_blocks,
    )

    model.train(epochs, loader, wandb=wandb)

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

    parser.add_argument('-m', '--memory')

    args = parser.parse_args()

    train_diffusion(
        checkpoint_path=args.checkpoint,
        memory_checkpoint_path=args.memory,
        device=args.device,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
