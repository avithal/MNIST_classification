import torch
from torch import nn
import pytorch_lightning as pl


# Step 1: Define the LightningModule for MNIST classification
class MNISTModel(pl.LightningModule):
    def __init__(self, model_params, optimizers=None):
        super().__init__()
        self.save_hyperparameters()
        if optimizers is None:
            self.learning_rate = 0.0001
            self.optimizer_type = "adam"  # Optimizer type (e.g., adam, sgd)
            self.weight_decay = .01
            self.lr_scheduler = None
        else:
            self.learning_rate = optimizers['learning_rate']
            self.optimizer_type = optimizers['optimizer']  # Optimizer type (e.g., adam, sgd)
            self.weight_decay = optimizers['weight_decay']
            self.lr_scheduler = optimizers['lr_scheduler']

        self.output_class = model_params['output_classes']

        # Define a simple fully connected neural network
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_class)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):

        if self.optimizer_type == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate,
                                    momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
        elif self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0005)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

        return [optim], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]

    def on_train_epoch_end(self):
        # Log the learning rate
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
