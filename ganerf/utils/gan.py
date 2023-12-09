"""
Generative Adversarial Network and its Components
"""
from typing import Literal, Optional, Set, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.printing import print_tcnn_speed_warning


def adversarial_loss(d_real, d_fake):
    """ Compute L_adv = 1/N * (f(D(fake)) + f(-D(real))),
        where f(x) = -log(1 + exp(-x)) 
    """
    l_fake = -torch.log(1 + torch.exp(-d_fake))
    l_real = -torch.log(1 + torch.exp(d_real))
    l_adv = (l_fake + l_real).mean()

    return l_adv


class Discriminator(FieldComponent):
    """Discriminator

    Args:
        in_dim: Input layer dimension
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        assert len(in_dim) == 3 and in_dim[0] == 3
        self.in_dim = in_dim

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        else :
            print_tcnn_speed_warning("This component doesn't support TCNN yet. Using torch implementation instead.")
            self.build_nn_modules()
    
    def build_nn_modules(self) -> None:
        """Initialize the discriminator components."""
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        downsampled_size = [self.in_dim[1]//8, self.in_dim[2]//8]
        self.fc = nn.Linear(downsampled_size[0] * downsampled_size[1] * 128, 1)

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with the discriminator.

        Args:
            in_tensor: Network input

        Returns:
            Discriminator network output
        """
        x = in_tensor
        x = self.downsample(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class Generator(FieldComponent):
    """Generator

    Args:
        in_dim: Input layer dimension
        out_dim: Output layer dimension. Uses layer_width if None.
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        out_dim: Optional[int] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        else :
            print_tcnn_speed_warning("This component doesn't support TCNN yet. Using torch implementation instead.")
            self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize the generator components."""
        self.fc = nn.Linear(self.in_dim, 4 * 4 * 128)
        self.upsample_and_generate = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with the generator.

        Args:
            in_tensor: Network input

        Returns:
            Generator network output
        """
        x = in_tensor
        x = self.fc(x)
        x = x.reshape((-1, 128, 4, 4))
        x = self.upsample_and_generate(x)
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
