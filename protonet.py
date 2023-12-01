# protonet.py

from Model import BaseICLModel, BaseICLTransformer
import torch
from abc import ABCMeta, abstractmethod

class ProtoNet(BaseICLModel, metaclass=ABCMeta):
    """
    Base class for ProtoNet models.

    Methods:
        _protonet_logits: Compute the ProtoNet logits for a given tensor of prototypes.
        _compute_features: Compute the ProtoNet embeddings for a given batch of features.
        forward: Compute the ProtoNet logits for a given batch of features.
    """
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
    

    @abstractmethod
    def _compute_features(self, x : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        """
        Compute the ProtoNet embeddings for a given batch of features.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
        """
        return
    

    def forward(self, features : torch.Tensor) -> torch.Tensor:
        """
        Computes protonet logits for a given batch of features, using the transformer
        encoder to compute the embeddings.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mask = torch.full(features.shape[:-1], False, device=self.device) 
        mask[:, -1, 1:] = True # mask out all but the first element of the query class

        predicted_features = self._compute_features(x=features, mask=mask)

        prototypes = torch.sum(predicted_features, dim=2) / torch.sum(~mask, dim=2)[..., None] # compute the prototypes for each class
        return self._protonet_logits(prototypes) # compute the protonet logits



class ProtoNetICL(BaseICLTransformer, ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder.
    """
    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(x, mask)
    
    

class ProtoNetSkip(BaseICLTransformer, ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder, but with
    a skip connection instead.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initialise the last layer of the encoder to zero, so that the skip connection is initialized to the identity
        self.encoder._init_last_layer_zero() 


    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x + self.encoder.forward(x, mask)