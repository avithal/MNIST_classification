import yaml
import random
import itertools

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from MODELS.LeNetModel import MNISTModel
from DATA.MNISTDataModule import MNISTDataModule
from utils import cleanup

# Load hyperparameters from YAML configuration
with open('MNIST_simple.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


def set_config_from_hyperparameters(config, hparams):
    config["training"]["learning_rate"] = hparams["learning_rate"]
    config["training"]["optimizer"] = hparams["optimizer"]
    config["training"]["weight_decay"] = hparams["weight_decay"]
    return config


def run_experiment(hparams, max_epochs):
    """Run a single training experiment with given hyperparameters."""
    datamodule = MNISTDataModule()
    #data_module.prepare_data() # download data from MNIST
    datamodule.setup_data()
    config = set_config_from_hyperparameters(CONFIG, hparams)
    model = MNISTModel(config['model'], config['training']
    )

    logger = CSVLogger(
        save_dir="logs",
        name="hyperparam_search"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="max",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, datamodule=datamodule)

    best_val_acc = checkpoint_callback.best_model_score
    return best_val_acc.item() if best_val_acc is not None else 0.0


def main(max_epochs=10,search_type = 'grid', num_trials = 10  ):
    """

    :param max_epochs: for each parameter
    :param search_type: choices=["grid", "random"]
    :param num_trials: Used only for random search
    :return:
    """

    # -------------------------------------------------
    # Hyperparameter space
    # -------------------------------------------------
    search_space = {
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64, 128],
        "optimizer": ["adam", "sgd"],
        "weight_decay": [0.0, 1e-4]
    }

    # -------------------------------------------------
    # Generate experiments
    # -------------------------------------------------
    if search_type == "grid":
        experiments = list(itertools.product(
            search_space["learning_rate"],
            search_space["batch_size"],
            search_space["optimizer"],
            search_space["weight_decay"]
        ))
    else:
        experiments = [
            (
                random.choice(search_space["learning_rate"]),
                random.choice(search_space["batch_size"]),
                random.choice(search_space["optimizer"]),
                random.choice(search_space["weight_decay"]),
            )
            for _ in range(num_trials)
        ]

    print(f"Running {len(experiments)} experiments...")

    best_score = 0.0
    best_config = None

    # -------------------------------------------------
    # Run search
    # -------------------------------------------------
    for i, (lr, batch_size, optimizer, wd) in enumerate(experiments):
        print(f"\nExperiment {i+1}/{len(experiments)}")
        print(f"lr={lr}, batch_size={batch_size}, optimizer={optimizer}, wd={wd}")

        hparams = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "weight_decay": wd
        }

        val_acc = run_experiment(hparams, max_epochs)
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_config = hparams
        cleanup()

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    print("\n==============================")
    print("Best configuration found:")
    for k, v in best_config.items():
        print(f"{k}: {v}")
    print(f"Best validation accuracy: {best_score:.4f}")
    print("==============================")


if __name__ == "__main__":
    pl.seed_everything(42)
    main(search_type="grid")
