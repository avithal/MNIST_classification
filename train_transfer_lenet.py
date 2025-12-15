from pytorch_lightning import Trainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from DATA.MNISTDataModule import MNISTDataModule
from MODELS.LeNetTransfer import LeNetTransfer

# TensorBoard logger saves logs under: tb_logs1/mnist_model
logger = TensorBoardLogger("tb_logs1", name="mnist_model")
# tensorboard  --logdir="D:\Avithal Study\MNIST_classification\tb_logs\mnist_model"


if __name__ == "__main__":
    # Example dataloaders
    # Load and prepare MNIST dataset
    data_module = MNISTDataModule()
    # data_module.prepare_data() # download data from MNIST
    data_module.setup_data()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    max_epochs = 25
    path_model = r'D:\Avithal Study\MNIST_classification\tb_logs1\mnist_model\version_1\checkpoint1\best_model-epoch=22-val_loss=0.0934.ckpt'

    model = LeNetTransfer(
        pretrained_path=path_model,
        new_num_classes=10,
        lr=1e-3
    )

    from pathlib import Path

    dirpath_save_checkpoints = Path(logger.log_dir) / "checkpoint1"

    # Create a checkpoint callback to save the best 3 models based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Monitor the validation loss
        dirpath=dirpath_save_checkpoints,  # Directory to save models
        filename='best_model-{epoch:02d}-{val_loss:.4f}',  # Filename format
        save_top_k=3,  # Save the top 3 models
        mode='min',  # Save the lowest validation loss (min)
        save_weights_only=True,  # Save only the model weights (not the entire model)
        verbose=True
    )

    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, accelerator='gpu',
                         callbacks=[checkpoint_callback], devices="auto")

    # Train the model
    # trainer.fit(model, train_loader,val_loader)
    trainer.fit(model, datamodule=data_module)

    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
