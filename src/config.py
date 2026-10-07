from pathlib import Path
import random
import numpy as np
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed):
    import torch
    if seed is None:
        raise ValueError("Set experiment.seed in the YAML before training.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
