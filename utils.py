import yaml
import torch
import numpy as np


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(self, key.upper(), value)