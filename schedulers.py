from torch.optim.lr_scheduler import LRScheduler
import math
from abc import abstractmethod, ABCMeta


class WarmupAndDecayLRScheduler(LRScheduler, metaclass=ABCMeta):
    """
    A base class for learning rate schedulers that incorporate warm-up and decay steps.

    Args:
        warmup_steps (int): The number of warm-up steps.
        *args: Additional positional arguments to be passed to the parent class.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    
    Attributes:
        warmup_steps (int): The number of warm-up steps.

    Inherits:
        LRScheduler: Base class for learning rate schedulers.

    Methods:
        _warmup_step(): Calculates a warm-up step for the learning rate scheduler.
        _decay_step(): Calculates a decay step for the learning rate scheduler.
        get_lr(): Get the learning rate for the current epoch (or step).
    """
    def __init__(self, warmup_steps, **kwargs) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(**kwargs)
    

    @abstractmethod
    def _warmup_step(self) -> list[float]:
        """
        Calculates a warm-up step for the learning rate scheduler.

        Returns:
            float: The warm-up step value.
        """
        return
    

    @abstractmethod
    def _decay_step(self) -> list[float]:
        """
        Calculates a decay step for the learning rate scheduler.
        
        Returns:
            float: The decay step value.
        """
        return


    def get_lr(self) -> list[float]:
        """
        Get the learning rate for the current epoch (or step).

        If the current epoch is less than the warm-up steps, the learning rate is calculated using the warm-up step function.
        Otherwise, the learning rate is calculated using the decay step function.

        Returns:
            list[float]: The learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            return self._warmup_step()
        else:
            return self._decay_step()
        


class InverseSqrtLR(WarmupAndDecayLRScheduler):
    """
    Learning rate scheduler that implements an inverse square root schedule
    with a linear warmup.

    Parameters:
        - d_model (int): The model dimension.
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

    Attributes:
        - scale_factor (float): The scale factor for the learning rate.

    Inherits:
        - WarmupAndDecayLRScheduler: Base class for learning rate schedulers that incorporate 
        warm-up and decay steps.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _warmup_step(self) -> list[float]:
        """
        A linear warmup step for the learning rate. The maximum learning rate is 
        equal to the learning rate provided to the optimizer.

        Returns:
        - float: The learning rate for the current warmup step.
        """
        scale = (self.last_epoch + 2) / (self.last_epoch + 1 * self.warmup_steps)
        return [x * scale for x in self.base_lrs]
    

    def _decay_step(self) -> list[float]:
        """
        An inverse square root decay step for the learning rate. The learning rate 
        is equal to the scale factor multiplied by the current epoch to the power of -1/2.

        Returns:
        - float: The learning rate for the current decay step.
        """
        scale = (self.last_epoch - self.warmup_steps + 2) / (self.last_epoch - self.warmup_steps + 1)
        return [x * (scale ** (-1/2)) for x in self.base_lrs]



class CosineAnnealingLRWithWarmup(WarmupAndDecayLRScheduler):
    """
    Learning rate scheduler that implements a cosine annealing schedule
    with a linear warmup.

    Parameters:
        - d_model (int): The model dimension.
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

    Attributes:
        - scale_factor (float): The scale factor for the learning rate.

    Inherits:
        - WarmupAndDecayLRScheduler: Base class for learning rate schedulers that incorporate 
        warm-up and decay steps.
    """
    def __init__(self, cosine_annealing_steps, min_lr=0, **kwargs):
        self.cosine_annealing_steps = cosine_annealing_steps
        self.min_lr = min_lr
        super().__init__(**kwargs)


    def _warmup_step(self) -> list[float]:
        """
        A linear warmup step for the learning rate. The maximum learning rate is 
        equal to the scale factor multiplied by the warmup steps to the power of -1/2.

        Returns:
        - float: The learning rate for the current warmup step.
        """
        return [x * (self.last_epoch / self.warmup_steps) for x in self.base_lrs]
    

    def _decay_step(self) -> list[float]:
        return [
            self.min_lr + (x - self.min_lr) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.cosine_annealing_steps)) / 2
            for x in self.base_lrs
            ]
    


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    def test():
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cos = CosineAnnealingLRWithWarmup(1000, 0.01, warmup_steps=1000, optimizer=optimizer)
        cos.last_epoch = 1500
        print(cos.get_lr())