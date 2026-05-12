import matplotlib.pyplot as plt


def evaluate_to_image(model, output_path) -> None:
    x = model.sample(16).to("cpu")
    x = (x + 1) / 2

    fig, axis = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(4):
        for j in range(4):
            axis[i, j].imshow(x[i * 4 + j].mean(dim=0).clip(0, 1))
            axis[i, j].grid(None)

    plt.savefig(output_path)