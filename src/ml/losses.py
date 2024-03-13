from typing import Any

import pytorch_metric_learning
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn as nn
import torch.optim as optimizer

class LossFunction(nn.Module):

    def __init__(self, loss_function) -> None:
        super().__init__()

        # Main Loss function
        self.loss_function = loss_function

    def forward(self, logits, targets) -> Any:
        
        # Get loss
        loss = self.loss_function(logits, targets)

        return loss

