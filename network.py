import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mlp, self).__init__()

        self.lc1 = nn.Linear(input_dim, 64)
        self.lc2 = nn.Linear(64, 64)
        self.lc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.lc1(x))
        x = F.relu(self.lc2(x))
        x = self.lc3(x)

        return x

        