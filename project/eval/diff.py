import argparse

from matplotlib import pyplot as plt

from project.eval.config import DiffEvalConfig
from project.models import EDMEvelynn
from project.util.evaluate import evaluate_to_image


def evaluate(
        checkpoint_path: str,
        device: str,
        output_path: str,
) -> None:

    config = DiffEvalConfig()

    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        model_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attn_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        network=config.network,
    ).to(device)

    print("Loading checkpoint...")
    model.load(checkpoint_path)
    print("Loaded checkpoint!")

    evaluate_to_image(model, output_path)


def main():

    parser = argparse.ArgumentParser(
        prog='Project eval (Diffusion)',
        description='This program manages eval for my BSc Thesis project (Diffusion).',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output')

    parser.add_argument('-d', '--device', default='cpu')

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
