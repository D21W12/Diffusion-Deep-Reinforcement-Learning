import argparse

from matplotlib import pyplot as plt

from project.eval.config import DiffEvalConfig
from project.models import EDMEvelynn

def evaluate(
        checkpoint_path: str,
        device: str,
        output_path: str,
) -> None:

    config = DiffEvalConfig()

    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        network=config.network,
    ).to(device)

    print("Loading checkpoint...")
    model.load(checkpoint_path)
    print("Loaded checkpoint!")

    x = model.heun_sampler(16).to("cpu")
    x = (x + 1) / 2

    fig, axis = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(4):
        for j in range(4):
            axis[i, j].imshow(x[i * 4 + j].permute(1, 2, 0).clip(0, 1))
            axis[i, j].grid(None)

    plt.savefig(output_path)


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
