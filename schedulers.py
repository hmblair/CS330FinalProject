from torch.optim.lr_scheduler import LRScheduler
   
class InverseSqrtLR(LRScheduler):
    """
    Learning rate scheduler that implements inverse square root decay, with linear warmup.

    Args:
        warmup_steps (int): The number of warmup steps for the learning rate.
        d_model (int): The model dimension.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        warmup_steps (int): The number of warmup steps for the learning rate.
        scale_factor (float): The scale factor for the learning rate calculation.

    Inherits:
        LRScheduler: Base class for learning rate schedulers.

    Methods:
        get_lr(): Get the learning rate for the current epoch.
    """
    def __init__(self, warmup_steps : int, d_model : int, *args, **kwargs):
        self.warmup_steps = warmup_steps # the number of warmup steps for the learning rate
        self.scale_factor = d_model ** (-1/2) # the model dimension
        super().__init__(*args, **kwargs)


    def get_lr(self) -> list[float]:
        """
        Get the learning rate for the current epoch.

        If the current epoch is less than the warmup steps, the learning rate is calculated via a linear warmup.
        Beyond this, the learning rate is calculated using an inverse square root decay..

        Returns:
            list: A list of learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            lr = self.scale_factor * self.last_epoch * (self.warmup_steps ** (-3/2))
        else:
            lr = self.scale_factor * (self.last_epoch ** (-1/2))
        return [lr for _ in self.base_lrs]