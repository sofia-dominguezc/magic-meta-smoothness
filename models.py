import math
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias) / math.sqrt(self.in_features)


class ScaledConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        kernel_dim = self.kernel_size[0] * self.kernel_size[1]
        fan_in = (self.in_channels // self.groups) * kernel_dim
        return self._conv_forward(input, self.weight, self.bias) / math.sqrt(fan_in)


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 10
    width: int = 16
    num_layers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class SimpleCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        
        in_c = config.in_channels
        out_c = config.width
        for i in range(config.num_layers):
            layers.append(ScaledConv2d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.MaxPool2d(2, 2))
            in_c = out_c
            out_c *= 2

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = ScaledLinear(in_c * 4 * 4, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
