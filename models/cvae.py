# https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple


class ConditionalVAE(nn.Module):

    def __init__(self, # TODO fix these arguments
                 in_channels: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: List = None,
                 board_shape: tuple[int] = (35,18),
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.board_shape = board_shape
        self.start_channels = in_channels

        # Embedding the non-board data (v_grade + angle)
        self.embed_metadata = nn.Linear(2, board_shape[0] * board_shape[1])

        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.last_hidden_dim = hidden_dims[-1]

        in_channels += 1 # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.MaxPool2d(2),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + 2, 1024)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1,
                                       ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.Upsample(scale_factor=2),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], stride=1, out_channels= self.start_channels,
                                      kernel_size = (4,3),
                                      padding = (3,2)
                                      ),
                            nn.Tanh())

    def encode(self, board_input: torch.tensor) -> List[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(board_input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.tensor) -> torch.tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.last_hidden_dim, 4, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (torch.tensor) Mean of the latent Gaussian
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, 
             board_data: torch.tensor, 
             y: torch.tensor, 
             **kwargs) -> List[torch.tensor]:

        embedded_metadata = self.embed_metadata(y)
        embedded_metadata = embedded_metadata.view(-1, self.board_shape[0], self.board_shape[1]).unsqueeze(1)

        embedded_input = self.embed_data(board_data)

        x = torch.cat([embedded_input, embedded_metadata], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim = 1)

        return  [self.decode(z), board_data, mu, log_var]

    def loss_function(self, recons, x, mu, log_var,
                      **kwargs) -> dict:

        kld_weight = kwargs['kld_weight']
        recons_loss = F.mse_loss(recons, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD':-kld_loss}

    def sample(self, y: torch.tensor,
               num_samples:int,
               current_device: int,
               **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor, y:torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x, y, **kwargs)[0]

