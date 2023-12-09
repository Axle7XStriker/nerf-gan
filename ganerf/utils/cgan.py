"""
Generative Adversarial Network and its Components
"""
from typing import Literal, Optional, Set, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.printing import print_tcnn_speed_warning

EMBEDDING_DIM = 100
LATENT_DIM=100
CONV_MIN_CHANNELS = 64


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
        self.label_condition_disc = nn.Sequential(
            nn.Embedding(num_embeddings=2, embedding_dim=EMBEDDING_DIM),
            nn.Linear(EMBEDDING_DIM, self.in_dim[0]*self.in_dim[1]*self.in_dim[2])
        )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=CONV_MIN_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=CONV_MIN_CHANNELS, out_channels=CONV_MIN_CHANNELS*2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*2, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=CONV_MIN_CHANNELS*2, out_channels=CONV_MIN_CHANNELS*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*4, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=CONV_MIN_CHANNELS*4, out_channels=CONV_MIN_CHANNELS*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*8, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear((self.in_dim[1]//16) * (self.in_dim[2]//16) * CONV_MIN_CHANNELS * 8, 1),
            nn.Sigmoid()
        )

    def pytorch_fwd(self, inputs: Tuple[Tensor]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with the discriminator.

        Args:
            in_tensor: Network input

        Returns:
            Discriminator network output
        """
        img, label = inputs
        label_output = self.label_condition_disc(label).view(-1, *self.in_dim)
        x = torch.cat((img, label_output), dim=1)
        x = self.model(x)
        return x

    def forward(self, inputs: Tuple[Tensor]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(inputs)
        return self.pytorch_fwd(inputs)


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
        assert len(in_dim) == 3 and in_dim[0] == 3 and len(out_dim) == 3
        self.in_dim = in_dim
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
        self.label_condition_gen = nn.Sequential(
            nn.Embedding(num_embeddings=2, embedding_dim=EMBEDDING_DIM),
            nn.Linear(EMBEDDING_DIM, 16)
        )

        self.final_conv_dim = (self.in_dim[1]//16, self.in_dim[2]//16)
        self.latent = nn.Sequential(
            nn.Linear(LATENT_DIM, self.final_conv_dim[0] * self.final_conv_dim[1] * CONV_MIN_CHANNELS*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=CONV_MIN_CHANNELS*8 + 1, out_channels=CONV_MIN_CHANNELS*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*8, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=CONV_MIN_CHANNELS*8, out_channels=CONV_MIN_CHANNELS*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=CONV_MIN_CHANNELS*4, out_channels=CONV_MIN_CHANNELS*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=CONV_MIN_CHANNELS*2, out_channels=CONV_MIN_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(CONV_MIN_CHANNELS),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=CONV_MIN_CHANNELS, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def pytorch_fwd(self, inputs: Tuple[Tensor]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with the generator.

        Args:
            in_tensor: Network input

        Returns:
            Generator network output
        """
        noise_vector, label = inputs
        label_output = self.label_condition_gen(label).view(-1, 1, *self.final_conv_dim)
        latent_output = self.latent(noise_vector).view(-1, CONV_MIN_CHANNELS*8, *self.final_conv_dim)
        x = torch.cat((latent_output, label_output), dim=1)
        x = self.model(x)
        return x

    def forward(self, inputs: Tuple[Tensor]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(inputs)
        return self.pytorch_fwd(inputs)


class ConditionalGAN(FieldComponent):
    """Conditional GAN

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
        self.generator = Generator(in_dim=in_dim, out_dim=out_dim)
        self.discriminator = Discriminator(in_dim=in_dim)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=1e-4, eps=1e-15, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, eps=1e-15, betas=(0.5, 0.999))
