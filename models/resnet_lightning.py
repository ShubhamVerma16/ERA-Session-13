import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from typing import Any

from utils.common import one_cycle_lr

class ResidualBlock(L.LightningModule):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.residual_block(x)

class ResNet(L.LightningModule):
    def __init__(
        self, batch_size=512, shuffle=True, num_workers=4, learning_rate=0.003, scheduler_steps=None, maxlr=None, epochs=None
    ):
        super(ResNet, self).__init__()
        self.data_dir = "./data"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.scheduler_steps = scheduler_steps
        self.maxlr = maxlr if maxlr is not None else learning_rate
        self.epochs = epochs

        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(channels=128),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(channels=512),
        )

        self.pool = nn.MaxPool2d(kernel_size=4)

        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = one_cycle_lr(
    optimizer=optimizer, maxlr=self.maxlr, steps=self.scheduler_steps, epochs=self.epochs
)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": "step"}}

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = nn.CrossEntropyLoss()(y_pred, y)

        preds = torch.argmax(y_pred, dim=1)

        accuracy = self.accuracy(preds, y)

        self.log_dict({"train_loss": loss, "train_acc": accuracy}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = nn.CrossEntropyLoss(reduction="sum")(y_pred, y)

        preds = torch.argmax(y_pred, dim=1)

        accuracy = self.accuracy(preds, y)

        self.log_dict({"val_loss": loss, "val_acc": accuracy}, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = nn.CrossEntropyLoss(reduction="sum")(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)

        accuracy = self.accuracy(preds, y)

        self.log_dict({"test_loss": loss, "test_acc": accuracy}, prog_bar=True)