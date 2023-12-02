from typing import Any
import torch
import torch.nn as nn

from base import BaseModel
from schedulers import InverseSqrtLR
from transformer_layers import Encoder


class BaseICLModel(BaseModel):
    """
    Base class for in-context meta-learning models.

    Args:
        objective (nn.CrossEntropyLoss): The cross entropy objective.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.objective = nn.CrossEntropyLoss() # the objective function


    def _compute_losses(self, batch: torch.Tensor) -> dict:
        """
        Compute the loss and accuracy for a given batch of data.

        Args:
            batch (torch.Tensor): The input batch of data, consisting of features and labels.

        Returns:
            dict: A dictionary containing the computed loss and accuracy.
        """
        features, labels = batch # unpack the batch
        query_labels = labels[:,-1] # get the query labels
        logits = self(features) # compute the logits
        loss = self.objective(logits, query_labels) # compute the loss for the query

        predictions = torch.argmax(logits, dim=1) # compute the predicted classes
        accuracy = torch.sum(predictions == query_labels) / torch.numel(predictions) # compute the accuracy

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




class BaseICLTransformer(BaseICLModel):
    """
    Base class for in-context meta-learning models that use transformers.

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
                 warmup_steps: int = 4000,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
        )
    

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get the scheduler for the model, which is an inverse square root scheduler with a linear warmup.
        This is the same scheduler used in 'Attention is All You Need'.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler object.
        """ 
        return InverseSqrtLR(optimizer=optimizer, warmup_steps=self.warmup_steps)