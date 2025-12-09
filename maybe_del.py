import torch
import pytorch_lightning as pl
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class MNISTModel(pl.LightningModule):
    def __init__(self, model_params, optimizers):
        super().__init__()
        self.learning_rate = optimizers['learning_rate']
        self.optimizer_type = optimizers['optimizer']
        self.weight_decay = optimizers['weight_decay']
        self.output_class = model_params['output_classes']

        # Define the model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_class)
        )
        self.criterion = nn.CrossEntropyLoss()

        # Track the best validation loss
        self.best_val_loss = float("inf")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"val_loss": loss}

    def on_validation_epoch_end(self, outputs):
        # Aggregate validation losses
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the average validation loss
        self.log("val_loss", avg_val_loss, prog_bar=True)

        # Check if this is the best validation loss
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.log("best_val_loss", self.best_val_loss, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        return optimizer

# Dataset and DataLoader for MNIST
def prepare_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=".", train=True, transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    # Define model parameters and optimizer settings
    model_params = {"output_classes": 10}
    optimizers = {"learning_rate": 1e-3, "optimizer": "adam", "weight_decay": 1e-5}

    # Initialize model, data, and trainer
    model = MNISTModel(model_params, optimizers)
    train_loader, val_loader = prepare_data()

    trainer = pl.Trainer(max_epochs=5, accelerator="auto", devices="auto")
    trainer.fit(model, train_loader, val_loader)
