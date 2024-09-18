import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def get_local_dir(local_dir: str) -> str:
    """Return the path to the cache directory for this user."""
    if os.path.exists(local_dir):
        return f"{local_dir}"
    os.makedirs(local_dir)
    return f"{local_dir}"


def get_local_run_dir(exp_name: str, local_dir: str) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    run_dir = f"{get_local_dir(local_dir)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

