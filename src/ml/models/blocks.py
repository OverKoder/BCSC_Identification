import torch.nn as nn

class ResidualLinearBlock(nn.Module):

  def __init__(self, in_features: int, out_features: int, activation: str):
    super().__init__()

    # Linear Layer
    self.linear = nn.Linear(in_features = in_features, out_features = out_features)

    # Activation function
    self.activation = nn.LeakyReLU()

    # BatchNorm
    self.batchnorm = nn.BatchNorm1d(num_features = out_features)

    return

  def forward(self, x):

    # Clone the tensor
    residual = x

    x = self.linear(x)

    # Add the residual connection
    x += residual

    x = self.activation(x)

    x = self.batchnorm(x)

    return x