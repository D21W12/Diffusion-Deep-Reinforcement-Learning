import argparse

import pandas as pd
import torch
from torchvision import transforms

from project.models import EDMMauMau, EDMLuomen
from project.util.data import ReplayMemoryData
from project.util.metrics import PSNR, MSE
from project.util.transforms import Difference

MODELS = {
    "maumau": EDMMauMau,
    "luomen": EDMLuomen,
}

def evaluate(memory, output, images, denoiser, sigma_noise) -> None:

    # Initializing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        Difference(),
        transforms.Normalize(0.5, 0.5),
    ])
    data = ReplayMemoryData(
        memory,
        transform,
        cap=images
    )

    # Initialize lists for metrics
    psnrs = []
    mses = []

    # Compute metrics for each image
    for i in range(images):

        y = data[i]
        y_noisy = y + sigma_noise * torch.randn_like(y)
        y_hat = denoiser.denoise(y_noisy)

        mses.append(MSE(y, y_hatt))
        psnrs.append(PSNR(y, y_hat))

    data = {
        "image": list(range(images)),
        "mse": mses,
        "psnr": psnrs,
        "sigma_noise": sigma_noise
    }
    df = pd.DataFrame(data)
    df.to_csv(output)

def main():

    parser = argparse.ArgumentParser(
        prog='Project eval (Denoising)',
        description='This program manages eval for my BSc Thesis project (Denoising).',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-m', '--memory', required=True)

    parser.add_argument('-o', '--output', required=True)

    parser.add_argument('-s', '--sigma', required=True, type=float)
    parser.add_argument('-n', required=True, type=int)

    parser.add_argument('-U', required=False, type=int)

    parser.add_argument('-d', '--device', default='cpu')

    args = parser.parse_args()

    model_ = args.model.lower()
    model = MODELS[model_]
    if model_ == "luomen":
        model = model.from_checkpoint(
            args.checkpoint,
            sigma_noise=args.sigma,
            U=args.U,
            device=args.device
        )
    elif model_ == "maumau":
        model = model.from_checkpoint(
            args.checkpoint,
            sigma_noise=args.sigma,
            device=args.device
        )

    evaluate(
        memory=args.memory,
        output=args.output,
        images=args.n,
        denoiser=model,
        sigma_noise=args.sigma,
    )


if __name__ == "__main__":
    main()
