# distributed_helper.py
import torch
from typing import Any
import warnings

def distributed_print(msg: Any) -> None:
    """
    Prints a message only once in a distributed setting, to avoid multiple
    processes printing the same message.

    Args:
        msg (Any): The message to be printed.
    """
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        and 
        torch.utils.data.get_worker_info() is None or torch.utils.data.get_worker_info().id == 0):
        print(msg)


def distributed_breakpoint() -> None:
    """
    Calls breakpoint() only once in a distributed setting, to avoid multiple
    processes calling breakpoint() at the same time.
    """
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        and 
        torch.utils.data.get_worker_info() is None or torch.utils.data.get_worker_info().id == 0):
        breakpoint()