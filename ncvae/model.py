import torch
from torch import nn
from torch.distributions import Normal, NegativeBinomial, Categorical, MixtureSameFamily, Independent

from .base import * 

class MixtureVAE(nn.Module):
    """Uses a mixture of Gaussians to approximate the latent space, as opposed to KL divergence for regularization. Acheives reasonable results."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64, n_gaussians: int = 10):
        super(VAE, self).__init__()
        # Message Passing
        self.message_sender = MessageSender(input_dim)
        self.message_receiver = MessageReceiver(input_dim)

        # Latent Representation Regularization
        self.n_gaussians = n_gaussians
        self.weights = nn.Parameter(torch.zeros(self.n_gaussians))
        self.means = nn.Parameter(torch.zeros(self.n_gaussians, latent_dim))
        self.logstds = nn.Parameter(torch.zeros(self.n_gaussians, latent_dim))

    def forward(self, x):
        z = self.message_sender.sample(x)
        alpha, beta = self.message_receiver(z)
        return z, alpha, beta

    def loss(self, x):
        z, alpha, beta = self.forward(x)
        beta += (x.sum(axis=1, keepdims=True) / x.sum(axis=1, keepdims=True).mean()).log()
        nb_nll = -NegativeBinomial(total_count=alpha, logits=beta).log_prob(X).sum(axis=1)
        mixing_distribution = Categorical(probs=torch.softmax(self.weights, dim=0))
        component_distribution = Independent(Normal(self.means, self.logstds.exp() + 1e-4), 1)
        gmm = MixtureSameFamily(mixing_distribution, component_distribution)
        latent_nll = -gmm.log_prob(z)

        return (latent_nll + nb_nll).mean()

class NCVAE(nn.Module):
    """VAE with Noise Contrastive Estimation and batch effect correction."""
    # STACK IT UP
    # STACK IT UP
    # STACK IT UP
    # STACK IT UP
    # STACK IT UP
    # STACK IT UP
    # STACK IT UP

    def __init__(self, input_dim: int, batch_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(NCVAE, self).__init__()
        self.message_sender = MessageSender(input_dim)
        self.message_receiver = MessageReceiver(input_dim)
        self.batch_effect_layer = nn.Linear(batch_dim, input_dim, bias=False)

    def forward(self, x, x_batch):
        z = self.message_sender.sample(x)
        alpha, beta = self.message_receiver(z)
        beta = beta + self.batch_effect_layer(x_batch)
        return z, alpha, beta

    def latent_representation(self, x):
        z, _, _ = self.forward(x, torch.zeros_like(x[:, :1]))  # dummy batch
        return z

    def loss(self, x, x_batch):
        z, alpha, beta = self.forward(x, x_batch)
        beta += (x.sum(axis=1, keepdims=True) / x.sum(axis=1, keepdims=True).mean()).log()
        nb_nll = -NegativeBinomial(total_count=alpha, logits=beta).log_prob(x).sum(axis=1)
        return nb_nll.mean()



class NCVAEDiscriminator(nn.Module):
    """Very simple discriminator module for the latent space"""
    
    def __init__(self, latent_dim: int = 64):
        super(NCVAEDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
