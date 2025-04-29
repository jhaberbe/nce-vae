import torch
from torch import nn
from torch.distributions import Normal, NegativeBinomial, Categorical, MixtureSameFamily, Independent

class Encoder(nn.Module):
    """Simple Encoder Module"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    """Simple Decoder Module"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MessageSender(nn.Module):
    """Defines the Encoder Head of the VAE. Includes functionality for sampling and returning latent representations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(MessageSender, self).__init__()

        self.mu_encoder = Encoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.logstd_encoder = Encoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    def forward(self, x):
        # TODO: Should probably be removed in the future
        log_x = torch.log1p(x)
        mu = self.mu_encoder(log_x)
        logstd = self.logstd_encoder(log_x)
        return mu, logstd

    def get_latent_representation(self, x):
        # TODO: Should probably be removed in the future
        log_x = torch.log1p(x)
        return self.mu_encoder(log_x)

    def sample(self, x):
        mu, logstd = self.forward(x)
        return Normal(mu, torch.exp(logstd) + 1e-4).rsample()


class MessageReceiver(nn.Module):
    """Defines the Decoder Head of the VAE. Returns the alpha and beta.""" 

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(MessageReceiver, self).__init__()

        self.alpha_decoder = Decoder(input_dim)
        self.beta_decoder = Decoder(input_dim)

    def forward(self, z):
        alpha = self.alpha_decoder(z)
        beta = self.beta_decoder(z)
        return F.softplus(alpha), beta
