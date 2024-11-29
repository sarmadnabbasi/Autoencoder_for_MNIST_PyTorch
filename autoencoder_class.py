import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoder_MNIST(nn.Module):
    def __init__(self, m=10):
        super(AutoEncoder_MNIST, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Input to hidden
            nn.ReLU(),
            nn.Linear(128, m)  # Hidden to bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(m, 128),  # Bottleneck to hidden
            nn.ReLU(),
            nn.Linear(128, 28 * 28),  # Hidden to output
            nn.Sigmoid()  # Output scaled to (0, 1)
        )

    def forward(self, x):
        z = self.encoder(x)  # Encode
        x_reconstructed = self.decoder(z)  # Decode
        return x_reconstructed, z


class VAE_MNIST(nn.Module):
    def __init__(self, m = 2, hidden_dim = 128):
        super(VAE_MNIST, self).__init__()
        self.m = m
        input_dim = 28*28

        # Encoder: Maps input to latent space parameters (mean and log variance)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, self.m)  # Mean layer
        self.logvar_layer = nn.Linear(hidden_dim, self.m)  # Log variance layer

        # Decoder: Maps latent space back to the input space
        self.decoder = nn.Sequential(
            nn.Linear(self.m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output scaled to (0, 1)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample noise
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

    # VAE Loss function: Reconstruction loss + KL divergence
    def loss_function(self, recon_x, x, mu, std):
        # Reconstruction loss (binary cross-entropy or MSE) using BCE as it is a probabilistic problem
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
        L = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2) # KL Divergence
        return BCE + L