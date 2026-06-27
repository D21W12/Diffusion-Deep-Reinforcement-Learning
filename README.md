# Using Diffusion Score-Matching to Reconstruct Observations in Deep Reinforcement Learning

> The thesis can be found in the `latex` submodule referred in this repository.

This repository contains the implementation of my BSc thesis: Using Diffusion Score-Matching to Reconstruct Observations in Deep Reinforcement Learning. The repository is based on the opensource Gymnasium framework for reinforcement learning environments, and implements all the algorithms from scratch using PyTorch. The project contains implementations alongside the proposed RePaint Heun sampler for EDM-based diffusion models.

## Installation

Step 1. Clone the repository using the following command

```
git clone git@github.com:D21W12/Improving-Deep-Reinforcement-Learning-using-Diffusion-Score-Matching.git
```

Step 2. Install PyTorch for your complute platform from the official (installation)[https://pytorch.org/get-started/locally/] page. The following installs `torch` and `torchvision` for cuda 1.13.2 using pip3.

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Step 3. Install the repositories requirements using the `requirements.txt`

```
pip3 install -r "requirements.txt"
```

Step 4. Download the checkpoints and datasets from the associated [huggingface dataset](https://huggingface.co/datasets/D21W12/Diffusion-Deep-Reinforcement-Learning)

Step 5. Done!

## Training

Aside from using the checkpoints available in the dataset, you can also train your own model, e.g., if you want to train it for a different environment. The project contains a CLI tool for training both the DQN and Diffusion Model. Training configurations can be adapted in `project/train/config.py`.

### DQN

To train the DQN one can run:

```
python3 -m project.train.dqn [-h] -c CHECKPOINT [-i INTERVAL] [-d DEVICE] -e EPOCHS --memory MEMORY -t TAG
```

### DDQN

The original experiments were performed using the DDQN loss, and can be trained using:

```
python3 -m project.train.ddqn [-h] -c CHECKPOINT [-i INTERVAL] [-d DEVICE] -e EPOCHS --memory MEMORY -t TAG
```

### Diffusion Model 

To train the Diffusion Model one can subsequently run:

```
python3 -m project.train.diff [-h] -c CHECKPOINT -d DEVICE -e EPOCHS -m MEMORY [-t TAG] [--dir DIR]
```

## Evaluating & Experiments

The models can be evaluated using the commands below. The evaluation and experiment config can be found in `project/eval/config.py`.

### DQN/DDQN

The (D)DQN agent can be evaluated, either manually or empiracally, using the following command:

```
python3 -m project.eval.dqn [-h] -c CHECKPOINT [-d DEVICE] [--manual] [-n NUMBER] [-o OUTPUT]
```

### Diffusion Model

The diffusion model can be sampled for 16 images using the following command:

```
python3 -m project.eval.diff [-h] -c CHECKPOINT [-o OUTPUT] [-d DEVICE]
```

### Experiments

The experiments can be run using the following command:

```
python3 -m project.eval.experiment [-h] --dqn DQN [--diffusion DIFFUSION] [-d DEVICE] [-s SETUP] [-n NOISE] [-m] [-o OUTPUT]
```

## Notebooks

All figures and metrics were computed using the notebooks in the `notebooks` folder. However, not all figures can be computed using the notebooks as they have been altered multiple times during the analysis of my project.

## LaTeX

The LaTeX submodule contains the the LaTeX source code of the thesis itself.
