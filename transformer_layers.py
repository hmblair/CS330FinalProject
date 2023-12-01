# transformer_layers.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers = {}
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = InterLeavedEncoderBlock(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            )
        self.layers = nn.ModuleDict(layers)


    def _freeze_layers(self) -> None:
        """
        Freeze the parameters of the encoder, mostly for testing purposes. 
        """
        for param in self[f"encoder_layer_{len(self.layers) - 1}"].parameters():
            param.requires_grad = False


    def _init_last_layer_zero(self) -> None:
        """
        Initialise the last layer of the encoder to zero.
        """
        for param in self.layers[f"encoder_layer_{len(self.layers) - 1}"].parameters():
            param.data.zero_()


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers.values():
            x = layer(x, mask)
        return x



class IntraClassEncoderBlock(nn.Module):
    # TODO: Fix dropout vs. attention_dropout
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            eps = 1E-6,
    ) -> None:
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
           d_model = hidden_dim,
           nhead=num_heads,
           dim_feedforward=mlp_dim,
           dropout=dropout,
           layer_norm_eps=eps,
           )


    def forward(self, input: torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        b,n,k,d = input.shape
        input = input.reshape((b*n,k,d))
        if mask is not None:
            mask = mask.reshape((k,b*n))    
        out = self.block(input, src_key_padding_mask=mask)
        return out.reshape((b,n,k,d))



class InterClassEncoderBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            eps=1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.block = nn.TransformerEncoderLayer(
           d_model = hidden_dim,
           nhead=num_heads,
           dim_feedforward=mlp_dim,
           dropout=dropout,
           layer_norm_eps=eps
           )

    def forward(self, input: torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        b,n,k,d = input.shape
        input = input.reshape((b,n*k,d))
        if mask is not None:
          mask = mask.reshape((n*k,b))
        out = self.block(input, src_key_padding_mask=mask)
        if mask is not None:
          return torch.sum(out.reshape((b,n,k,d)), dim=2) / torch.sum(~mask.reshape((b,n,k)), dim=2)[..., None]
        else:
           return torch.mean(out.reshape((b,n,k,d)), dim=2)
        


class InterLeavedEncoderBlock(nn.Module):
    """
    Implements a single inter- and intra-class transformer layer. 

    Args:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension size.
        mlp_dim (int): MLP dimension size.
        dropout (float): Dropout rate.
        attention_dropout (float): Attention dropout rate.
    """
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        ) -> None:

        super().__init__()
        self.intra_block = IntraClassEncoderBlock(
           num_heads=num_heads,
           hidden_dim=hidden_dim,
           mlp_dim=mlp_dim,
           dropout=dropout,
           attention_dropout=attention_dropout
        )
        
        self.inter_block = InterClassEncoderBlock(
           num_heads=num_heads,
           hidden_dim=hidden_dim,
           mlp_dim=mlp_dim,
           dropout=dropout,
           attention_dropout=attention_dropout)
        

    def forward(self, input : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InterLeavedEncoderBlock.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, way, shot, feature_dim).
            mask (torch.Tensor): Mask tensor of shape (batch_size, way, shot, feature_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, way, shot, feature_dim).
        """
        x = self.intra_block(input, mask)
        y = self.inter_block(x, mask)
        return x + y[:, :, None, :]
  