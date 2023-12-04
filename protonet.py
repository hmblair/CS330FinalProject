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


    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity between two tensors.

        Args:
            x (torch.Tensor): The first tensor.
            y (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The cosine similarity between the two tensors.
        """
        return torch.sum(x * y, dim=-1) / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))


    def _protonet_logits(self, support, query : torch.Tensor) -> torch.Tensor:
        """
        Compute the ProtoNet logits for a given tensor of prototypes. The last
        element of each batch is the query, and all other elements are classes.

        Args:
            x (torch.Tensor): The input batch of data, consisting of features and labels.

        Returns:
            torch.Tensor: The ProtoNet logits.
        """
        return self._cosine_similarity(query[:, None], support) # compute the negative squared distances between the query and each class
    

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

        support_features = predicted_features[:, :-1] # the support features are all but the last element of each batch
        query_features = predicted_features[:, -1, 0] # the query features are the last element of each batch and the first shot only
        support_prototypes = torch.mean(support_features, dim=2) # compute the prototypes for each class

        # prototypes = torch.sum(predicted_features, dim=2) / torch.sum(~mask, dim=2)[..., None] # compute the prototypes for each class
        return self._protonet_logits(support_prototypes, query_features) # compute the protonet logits




class ProtoNetICL(BaseICLTransformer, ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder.
    """
    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(x, mask)
    



class ProtoNetWithoutEncoder(ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder.
    """
    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x
    


class ProtoNetLinear(ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    


class ProtoNetSkip(BaseICLTransformer, ProtoNet):
    """
    ProtoNet with in-context learning provided by a transformer encoder, but with
    a skip connection instead.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def _compute_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x + self.encoder.forward(x, mask)