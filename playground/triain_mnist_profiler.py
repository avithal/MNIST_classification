from MODELS.MNISTModel import MNISTModel
from DATA.MNISTDataModule import MNISTDataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
import gc
from pytorch_lightning.profilers import PyTorchProfiler
from utils import cleanup
# TensorBoard logger saves logs under: tb_logs/mnist_model
logger = TensorBoardLogger("../tb_logs", name="mnist_model")
# tensorboard  --logdir="D:\Avithal Study\MNIST_classification\tb_logs\mnist_model"


if __name__ == '__main__':

    # Load hyperparameters from YAML configuration
    with open('../MNIST_simple.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Lightning model
    model = MNISTModel(config['model'], config['training'])

    # Load and prepare MNIST dataset
    data_module = MNISTDataModule()
    #data_module.prepare_data() # download data from MNIST
    data_module.setup_data()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # # Move model to GPU manually pytorch lightning does it automatically
    # model.to(device=device)


    # Directory path for checkpoints
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
    )    # https://ui.perfetto.dev




    # Lightning trainer with GPU + profiler enabled
    profiler = PyTorchProfiler(
        dirpath="lightning_logs12/profiler",
        filename="profiler_output",
        export_to_chrome=True,  # THIS is key
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    # https://ui.perfetto.dev

    trainer = pl.Trainer(max_epochs=config['training']['max_epochs'], logger=logger, accelerator='gpu', callbacks=[checkpoint_callback], devices=1, profiler=profiler)

    # Train the model
    #trainer.fit(model, train_loader,val_loader)
    trainer.fit(model, datamodule =data_module)

    # Copy hyperparameters
    import shutil
    from pathlib import Path
    config_path = Path("../MNIST_simple.yaml")
    target_dir = Path(logger.log_dir)
    shutil.copy(config_path, target_dir / "config.yaml")

    # Cleanup
    gc.collect()

    # Run test evaluation

    # Lightning trainer with GPU + profiler enabled
    test_profiler = PyTorchProfiler(
        dirpath="lightning_logs12/profiler",
        filename="test_output",
        export_to_chrome=True,  # THIS is key
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    trainer.test(model, data_module, profiler =test_profiler )

    # cleanup
    cleanup(trainer=trainer, model=model, datamodule=data_module, logger=logger)

