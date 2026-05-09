import argparse
import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from project.models import EDMEvelynn
from project.train.config import DiffTrainingConfig
from project.util.data import ReplayMemoryData


def train_diffusion(
        checkpoint_path: str,
        memory_checkpoint_path: str,
        device: str,
        epochs: int,
) -> None:

    config = DiffTrainingConfig(
        checkpoint_path=checkpoint_path,
        data_path=memory_checkpoint_path,
        device=device,
        epochs=epochs,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Pad(2)
    ])

    data = ReplayMemoryData(
        memory=config.data_path,
        transform=transform,
    )
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)

    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        batch_size=config.batch_size,
        lr=config.lr,
        network=config.network
    ).to(config.device)

    if os.path.exists(config.checkpoint_path):
        print("Loading checkpoint...")
        model.load(config.checkpoint_path)
        print("Loaded checkpoint!")

    model.train(config.epochs, loader)

    print("Saving checkpoint...")
    model.save(config.checkpoint_path)
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

    kwargs = {
        "model": args.model,
        "epochs": args.epochs,
        "device": args.device,
        "checkpoint_path": args.checkpoint,
        "memory_checkpoint_path": args.memory
    }

    train_diffusion(**kwargs)

if __name__ == "__main__":
    main()
