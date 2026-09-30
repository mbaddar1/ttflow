from torch import nn
import torch
from loguru import logger
# ✅ Model Definition
class VelocityFieldRegressionNet(nn.Module):
    """
    Name taken from here
    https://colab.research.google.com/github/drscotthawley/blog/blob/main/extra/FlowModels_colab.ipynb#scrollTo=6a22fdf8-2b73-4045-a46a-4459ad3b4888&line=4&uniqifier=1
    class VelocityNet(nn.Module):
        def __init__(self, input_dim, h_dim=64):
        .....
    also arch. double-checked against this colab notebook
    """

    def __init__(
            self, input_dim, hidden_dim, depth, output_dim=1, device=None, data_type=None
    ):
        """input_dim: Number of input features

        hidden_dim: Width of each hidden layer
        depth: Number of hidden layers
        output_dim: Size of output (1 for regression)
        device: Target device string or torch.device ('cuda', 'cpu', etc.)
        """
        super(VelocityFieldRegressionNet, self).__init__()

        # Auto-detect CUDA if device is not explicitly passed
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        data_type = torch.float32 if data_type is None else data_type
        layers = []

        # First layer from input to hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

        # Move the entire network to the target device
        self.to(device=self.device, dtype=data_type)

        # Inspect the first parameter's device
        actual_device = next(self.parameters()).device
        logger.info(f"Network is on: {actual_device}")

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
