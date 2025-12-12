import torch
from torch import nn
import pytorch_lightning as pl

from MODELS.ModelsGeneral import lenet_batchnorm

def clean_state_dict_for_sequential(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_k = k.replace("model.", "", 1)  # remove only the first "model."
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd

class LeNetTransfer(pl.LightningModule):
    def __init__(self, pretrained_path: str, new_num_classes: int = 10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # --- Load pretrained ---
        self.model = lenet_batchnorm(num_classes=10)
        state = torch.load(pretrained_path, map_location="cpu")
        state_model = clean_state_dict_for_sequential(state['state_dict'])
        self.model.load_state_dict(state_model)
        print("Loaded pretrained LeNet!")

        # --- Freeze early layers ---
        # Freeze conv1 + pool + conv2 + pool
        for param in self.model[:6].parameters():
            param.requires_grad = False

        # --- Replace classifier ---
        self.model[-1] = nn.Linear(84, new_num_classes)

        # --- Loss ---
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Train only unfrozen layers
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(trainable, lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
