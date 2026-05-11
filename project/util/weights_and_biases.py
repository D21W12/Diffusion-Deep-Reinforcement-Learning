import wandb


class WandB:

    def __init__(
            self,
            project: str,
            *args,
            **kwargs
    ):
        self._run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="daan-wichmann-radboud-university",
            # Set the wandb project where this run will be logged.
            project=project,
            # Track hyperparameters and run metadata.
            config=kwargs,
        )

    def log(self, *args, **kwargs):
        self._run.log(kwargs)

    def finish(self):
        self._run.finish()
