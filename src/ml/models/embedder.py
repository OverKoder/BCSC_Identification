from typing import Any

from blocks import ResidualLinearBlock

import torch.nn as nn

class Embedder(nn.Module):

    def __init__(self,
      input_size: int,
      layers: list,
      use_residual: bool, # If True makes each layer have a residual connection to avoid vanishing gradient
      prefix: str = '',
      suffix: str = ''
    ):
        super().__init__()

        # Main attributes

        self.input_size = input_size
        self.emb_size = layers[-1]

        # List of layers (architecture of the model). In this specific model, the number of neurons of last layer is taken as the
        # embedding size
        self.layers_list = layers
        
        self.use_residual = use_residual

        # Layers
        self.layers = []

        # Used to save the weights in a .pt fiel
        self.prefix = prefix
        self.suffix = suffix

        prev_features = self.input_size

        # If True the build layers with residual connections
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
                self.layers.append(nn.BatchNorm1d(num_features = layer))

                prev_features = layer

        # Unpack layers
        self.layers = nn.Sequential(*self.layers)

        return
    
    def get_params(self):
       
       params = {
          'input_size': self.input_size,
          'layers': self.layers_list,
          'use_residual': self.use_residual
       }

       return params
    
    def get_name(self):

      # Returns a string name to save the weights in a .pt file
      return self.prefix + str(len(self.layers_list)) + '_layers_' + str(self.emb_size) + '_' + self.suffix

    def forward(self, x) -> Any:
        return self.layers(x)