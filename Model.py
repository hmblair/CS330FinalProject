import torch
import torch.nn as nn

from base import BaseModel
from schedulers import InverseSqrtLR


class BaseICLModel(BaseModel):
    """
    Base class for in-context meta-learning models.

    Args:
        num_layers (int): The number of transformer layers in the model.
        num_heads (int): The number of attention heads in each transformer.
        hidden_dim (int): The hidden dimension.
        mlp_dim (int): The dimensionality of the MLP layers in the transforer layers.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        attention_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        hidden_dim (int): The hidden dimension.
        encoder (Encoder): The transformer encoder module of the model.
        objective (nn.CrossEntropyLoss): The objective (loss) function.

    Methods:
        _compute_losses(batch: torch.Tensor) -> dict:
            Compute the losses and accuracy for a given batch of data.

        _get_inputs_and_outputs(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            Get the inputs and outputs from a batch. Required for making predictions with the `BaseModel` class.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
        )

        self.objective = nn.CrossEntropyLoss() # the objective (loss) function


    def _compute_losses(self, batch: torch.Tensor) -> dict:
        """
        Compute the losses and accuracy for a given batch of data.

        Args:
            batch (torch.Tensor): The input batch of data, consisting of features and labels.

        Returns:
            dict: A dictionary containing the computed loss and accuracy.
        """
        features, labels = batch # unpack the batch
        logits = self(features) # compute the logits
        loss = self.objective(logits, labels[:,-1]) # compute the loss for the query

        predictions = torch.argmax(logits, dim=1) # compute the predicted classes
        accuracy = torch.sum(predictions == labels[:,-1]) / torch.numel(labels) # compute the accuracy

        return {'loss': loss, 'accuracy': accuracy}


    def _get_inputs_and_outputs(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the inputs and outputs from a batch. Required for making predictions with
        the `BaseModel` class.

        Args:
            batch (torch.Tensor): The input batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the inputs and outputs.
        """
        x, y = batch
        return x, y
    

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model, which is Adam.

        Returns:
            torch.optim.Optimizer: The Adam optimizer object.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get the scheduler for the model, which is an inverse square root scheduler with a linear warmup.
        This is the same scheduler used in 'Attention is All You Need'.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler object.
        """ 
        return InverseSqrtLR(optimizer=optimizer, warmup_steps=4000, d_model=self.hidden_dim)
  

class ProtoNet(BaseICLModel):
    def _protonet_logits(self, prototypes : torch.Tensor) -> torch.Tensor:
        """
        Compute the ProtoNet logits for a given tensor of prototypes. The last
        element of each batch is the query, and all other elements are classes.

        Args:
            x (torch.Tensor): The input batch of data, consisting of features and labels.

        Returns:
            torch.Tensor: The ProtoNet logits.
        """
        query = prototypes[:, -1] # the query is the last element of each batch
        classes = prototypes[:, :-1] # the classes are all but the last element of each batch
        return -torch.sum( (query[:, None] - classes) ** 2, dim = -1) # compute the negative squared distances between the query and each class


class ProtoNetICL(ProtoNet):
    def forward(self, features : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mask = torch.full(features.shape[:-1], False, device=self.device) 
        mask[:, -1, 1:] = True # mask out all but the first element of the query class

        predicted_features = self.encoder.forward(x=features, mask=mask) # pass the features through the encoder
        prototypes = torch.sum(predicted_features, dim=2) / torch.sum(~mask, dim=2)[..., None] # compute the prototypes for each class
        return self._protonet_logits(prototypes) # compute the protonet logits



class ProtonetClip(ProtoNet):
    def forward(self, features : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        """
        Computes the ProtoNet logits directly from the CLIP embeddings, without
        passing them through the transformer encoder. Useful for comparision. 

        Args:
            x (torch.Tensor): Input tensor.s
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        prototypes = torch.sum(features, dim=2) / torch.sum(~mask, dim=2)[..., None] # compute the prototypes for each class
        return self._protonet_logits(prototypes)



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
  