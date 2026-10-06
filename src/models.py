import torch.nn as nn
from torch import Tensor

# Both models use the standard parametrization with LeCun init (W ~ N(0, 1/fan_in), b = 0),
# which keeps (pre-)activations O(1) at every layer so that one global learning rate is
# reasonable for all layers (at a fixed width). Seed with torch.manual_seed before creating.


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 32, depth: int = 1):
        super().__init__()
        dims = [in_dim] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(dims[-1], out_dim)
        for layer in [*self.hidden, self.out]:
            nn.init.normal_(layer.weight, std=layer.in_features**-0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        for layer in self.hidden:
            x = nn.functional.gelu(layer(x))
        return self.out(x)


class CNN(nn.Module):
    """`depth` blocks of conv3x3 -> GELU -> maxpool (doubling the channels), then a linear head."""

    def __init__(self, in_channels: int, out_dim: int, width: int = 16, depth: int = 2):
        super().__init__()
        channels = [in_channels] + [width * 2**i for i in range(depth)]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(channels, channels[1:]))
        self.out = nn.Linear(channels[-1] * 4 * 4, out_dim)
        for layer in [*self.convs, self.out]:
            nn.init.normal_(layer.weight, std=layer.weight[0].numel() ** -0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = nn.functional.max_pool2d(nn.functional.gelu(conv(x)), 2)
        return self.out(nn.functional.adaptive_avg_pool2d(x, 4).flatten(1))
