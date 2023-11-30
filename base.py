from typing import Any
import torch
import pytorch_lightning as pl
from typing import Optional
import math
import warnings
from distributed_helper import distributed_print


class BaseModel(pl.LightningModule):
    """
    Base class for PyTorch Lightning models that abstracts away some of the boilerplate code.

    Args:
        lr (float, optional): The learning rate for the optimizer.

    Attributes:
        lr (float | None): The learning rate for the optimizer.
        objective (torch.nn.MSELoss): The loss function used for training.
    """
    def __init__(self, lr: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters()  # save the hyperparameters

        if lr is None:
            warnings.warn('No learning rate was provided. The model will not be able to be trained.')
        self.lr = lr  # the learning rate
        self.objective = torch.nn.MSELoss()  # the loss function


    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The output of the model.
        """
        return super().forward(*args, **kwargs)
    

    def _weight_init(self) -> None:
        """
        Initializes the model using Xavier initialization.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.ndim == 1:
                    param.data.fill_(0)
                else:
                    bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                    param.data.uniform_(-bound, bound)


    def training_step(self, batch : torch.Tensor, batch_ix : list[int]) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_ix (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the training step.
        """
        loss = self._compute_and_log_losses(batch, 'train') # compute the losses
        lr = self._get_lr() # get the learning rate
        self.log('lr', lr, prog_bar=True, on_step=True, sync_dist=True) # log the learning rate
        return loss
    

    def validation_step(self, batch : torch.Tensor, batch_ix : list[int]) -> None:
        """
        Perform a validation step on a batch of data.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_ix (int): The index of the batch.
        """
        _ = self._compute_and_log_losses(batch, 'val') # compute the losses

    
    def test_step(self, batch : torch.Tensor, batch_ix : list[int]) -> None:
        """
        Perform a test step on a batch of data.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_ix (int): The index of the batch.
        """
        _ = self._compute_and_log_losses(batch, 'test') # compute the losses


    def predict_step(self, batch : torch.Tensor, batch_ix : list[int]) -> torch.Tensor:
        """
        Perform a prediction step on a batch of data.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_ix (int): The index of the batch.

        Returns:
            torch.Tensor: The predicted outputs from the model 
            for the batch.
        """ 
        # HOW TO LOG THE LOSSES IN AN EFFECTIVE MANNER?
        x, _ = self._get_inputs_and_outputs(batch) # get the input from the batch
        return self(x) # return the predictions


    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler for the model.
        
        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler, and monitor
            for the scheduler.
        """
        if self.lr is None:
            raise ValueError('No learning rate was provided during initialization. The model cannot be trained.')
        optimizer = self._get_optimizer() # get the optimizer
        scheduler = self._get_scheduler(optimizer) # get the scheduler
        if scheduler is None:
            return {'optimizer': optimizer}
        else:
            scheduler_dict = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return {'optimizer': optimizer,'lr_scheduler': scheduler_dict}
        

    def on_train_epoch_start(self) -> None:
        """
        Print a new line at the start of each training epoch, to separeate the correspdonding
        progress bars.
        """
        distributed_print('\n')
    

    def _compute_and_log_losses(self, batch : torch.Tensor, phase : str) -> torch.Tensor:
        """
        Compute the relevant losses and log them, returning the loss that is 
        required for training.

        Args:
            batch (torch.Tensor): The input batch of data.
            phase (str): The current phase.

        Returns:
            torch.Tensor: The primary loss value for the current step.
        """
        losses = self._compute_losses(batch) # compute the losses
        for name, value in losses.items():
            self._validate_losses(value, name) # ensure that the loss is valid
            self._log(phase + '_' + name, value) # log the loss
        return losses['loss']


    def _compute_losses(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the losses for the model. The loss named 'loss' will be the one which
        is used to train the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed
            losses and their respective names.
        """
        raise NotImplementedError('The _compute_loss method must be implemented.')
    

    def _log(self, name: str, value: torch.Tensor, **kwargs) -> None:
        """
        Logs the given name-value pair with additional optional keyword arguments.

        Args:
            name (str): The name of the value being logged.
            value (torch.Tensor): The value to be logged.
            **kwargs: Additional optional keyword arguments.

        Returns:
            None
        """
        self.log(name, value, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False, **kwargs)

    
    def _validate_losses(self, loss : torch.Tensor, name : str) -> None:
        """
        Validates the loss value to ensure it is not NaN, infinite, or negative.

        Args:
            loss: The loss value to be validated.
            name: The name of the loss value.

        Raises:
            ValueError: If the loss value is NaN or infinite.
        """
        if loss.isnan():
            raise ValueError(f'The {name} is NaN.')
        if loss.isinf():
            raise ValueError(f'The {name} is infinite.')
        if loss < 0:
            warnings.warn(f'The {name} is negative.')


    def _get_inputs_and_outputs(self, batch : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the inputs and corresponding outputs from the given batch.

        Args:
            batch (torch.Tensor): The input batch.

        Returns:
            tuple(torch.Tensor, torch.Tensor): The inputs and corresponding outputs.
        
        Examples:
            >>> x, y = self._get_inputs_and_outputs(batch)
        """
        raise NotImplementedError('The _get_inputs_and_outputs method must be implemented.')
    

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        Gets the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        raise NotImplementedError('The _get_optimizer method must be implemented.')
        

    def _get_scheduler(self, optimizer : torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Gets the learning rate scheduler for the model.

        Returns:
            scheduler | None: The learning rate scheduler. If None, no scheduler will be used.
        """
        return None
        
    
    def _get_lr(self) -> float:
        """
        Retrieves the current learning rate.

        Returns:
            float: The current learning rate.
        """
        return self.optimizers().param_groups[0]["lr"]
    








from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
class BaseDataModule(pl.LightningDataModule, metaclass = ABCMeta):
    """
    Base class for creating data modules in PyTorch Lightning that abstracts away some of the boilerplate code.

    Args:
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for data loading. Default is 0.

    Attributes:
        data (dict): A dictionary to store the datasets for different phases.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for data loading.

    Methods:
        _create_datasets: Create datasets for the specified phase. This is an abstract method and must be implemented.
        setup: Sets up the data for the specified stage.
        train_dataloader: Returns the train dataloader.
        val_dataloader: Returns the validation dataloader.
        test_dataloader: Returns the test dataloader.
        pred_dataloader: Returns the prediction dataloader.
        _create_dataloaders: Create a dataloader for the specified phase.
    """
    def __init__(self, batch_size : int, num_workers : int = 0):
        super().__init__()
        self.data = {}
        self.batch_size = batch_size
        self.num_workers = num_workers


    @abstractmethod
    def _create_datasets(self, phase : str) -> torch.utils.data.Dataset:
        """
        Create datasets for the specified phase.

        Args:
            phase (str): The phase for which to create the datasets. Can be one of 'train', 'val', 'test', or 'predict'.

        Returns:
            torch.utils.data.Dataset: The dataset for the specified phase.
        """
        return   
    

    def setup(self, stage: str) -> None:
        """
        Sets up the data for the specified stage.

        Parameters:
        - stage (str): The stage of the data setup. Must be either 'fit', 'test', or 'predict'.

        Raises:
        - ValueError: If the stage is not one of 'fit', 'test', or 'predict'.
        """
        if stage == 'fit':
            self.data['train'] = self._create_datasets('train')
            self.data['val'] = self._create_datasets('val')
        elif stage in ['test', 'predict']:
            self.data[stage] = self._create_datasets(stage)
        else:
            raise ValueError('The stage must be either "fit", "test" or "predict".')


    def train_dataloader(self):
        """
        Returns the train dataloader.

        Returns:
            torch.utils.data.DataLoader: The train dataloader.
        """
        return self._create_dataloaders('train')


    def val_dataloader(self):
        """
        Returns the validaiton dataloader.

        Returns:
            torch.utils.data.DataLoader: The validation dataloader.
        """
        return self._create_dataloaders('val')


    def test_dataloader(self):
        """
        Returns the test dataloader.

        Returns:
            torch.utils.data.DataLoader: The test dataloader.
        """
        return self._create_dataloaders('test')


    def pred_dataloader(self):
        """
        Returns the prediction dataloader.

        Returns:
            torch.utils.data.DataLoader: The prediction dataloader.
        """
        return self._create_dataloaders('predict')     


    def _create_dataloaders(self, phase: str):
        """
        Create a dataloader for the specified phase.

        Args:
            phase (str): The phase for which to create the dataloaders. Can be one of 'train', 'val', 'test', or 'predict'.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the specified phase.
        """
        if phase not in ['train', 'val', 'test', 'predict']:
            raise ValueError(f'Unknown phase {phase}. Please specify one of "train", "val", "test", or "predict".')

        dataloader = DataLoader(
            dataset = self.data[phase],
            num_workers = self.num_workers,
            batch_size = self.batch_size,
        )

        return dataloader