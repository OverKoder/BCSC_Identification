from typing import Any, Optional

from blocks import ResidualLinearBlock

import torch
from torch.optim import AdamW, Optimizer
from torch import sigmoid
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import Accuracy, AUROC, BinaryPrecisionRecallCurve, BinaryF1Score


class MLP(pl.LightningModule):

    def __init__(self,
        input_size: int, 
        layers: list, 
        optimizer: Optimizer, 
        loss_function, 
        embedder: nn.Module = None, 
        use_residual: bool = False
        ):

        super().__init__()

        # Main attributes
        # List of layers (architecture of the model)
        self.layers_list = layers

        # Optimizer
        self.optimizer = optimizer

        # Loss
        self.loss_function = loss_function
        
        # Whether to use residual (skip) connections or not
        self.use_residual = use_residual

        # Layers
        self.layers = []

        prev_features = input_size

        # Use residual connection
        if self.use_residual:

            for layer in self.layers_list:

                if prev_features == layer:

                    # Append residual block
                    self.layers.append(
                        ResidualLinearBlock(in_features = prev_features, out_features = layer, activation = self.activation)
                    )

                else:
                    # Append linear layer
                    self.layers.append(nn.Linear(in_features = prev_features, out_features = layer))

                    # Activation function
                    self.layers.append(nn.LeakyReLU())

                    # BatchNorm
                    self.layers.append(nn.BatchNorm1d(num_features = layer))

                prev_features = layer

        else:
            for layer in self.layers_list:

                # Append linear layer
                self.layers.append(nn.Linear(in_features = prev_features, out_features = layer))

                # Activation function
                self.layers.append(nn.LeakyReLU())

                # BatchNorm
                #self.layers.append(nn.BatchNorm1d(num_features = layer))
                
                prev_features = layer

        self.layers.append(nn.Linear(in_features = prev_features, out_features = 1))

        # Unpack layers
        self.layers = nn.Sequential(*self.layers)

        # Embedder in case its used
        self.embedder = embedder

        # Metrics
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

        return

    def forward(self, x) -> Any:
        # The embedder's weights must not be changed with gradient,
        # so we wrap the embedder's computations with no_grad() to make sure
        # the gradient does not propagate to the embedder
        if self.embedder is not None:
            #with torch.no_grad():
            x = self.embedder(x)

            # Now, this tensor does need gradient, so we must activate it
            #x.requires_grad_()

        return self.layers(x)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch

        # Forward
        logits = self(inputs)

        # Compute loss and log it
        loss = self.loss_function(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        train_f1 = self.train_f1(logits, targets)
        self.log('training_f1', train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Forward
        logits = self(inputs)

        # Compute loss and log it
        loss = self.loss_function(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        val_f1 = self.val_f1(logits, targets)
        self.log('validation_f1', val_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = 1e-3)



class LinearMLP(pl.LightningModule):

    def __init__(self,
        input_size: int, 
        layers: list, 
        optimizer: Optimizer, 
        loss_function, 
        ):

        super().__init__()

        # Main attributes
        
        self.input_size = input_size

        # List of layers (architecture of the model)
        self.layers_list = layers

        # Optimizer
        self.optimizer = optimizer

        # Loss
        self.loss_function = loss_function

        # Layers
        self.layers = []

        prev_features = input_size

        for layer in self.layers_list:
            
            # No activation function
            self.layers.append(nn.Linear(in_features = prev_features, out_features = layer))

            prev_features = layer

        self.layers.append(nn.Linear(in_features = prev_features, out_features = 1))

        # Unpack layers
        self.layers = nn.Sequential(*self.layers)

        # Metrics
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

        return

    def forward(self, x) -> Any:
        return self.layers(x)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch

        # Forward
        logits = self(inputs)

        # Compute loss and log it
        loss = self.loss_function(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        train_f1 = self.train_f1(logits, targets)
        self.log('training_f1', train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Forward
        logits = self(inputs)

        # Compute loss and log it
        loss = self.loss_function(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        val_f1 = self.val_f1(logits, targets)
        self.log('validation_f1', val_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = 1e-3)