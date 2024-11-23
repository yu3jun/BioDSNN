import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import random

class VAE(nn.Module):
    def __init__(self, input_dim = 5045, inter_dim=32, latent_dim=16):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),  
            nn.ReLU(),
            nn.Linear(2048, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, inter_dim),  
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(inter_dim, latent_dim)
        self.fc_var = nn.Linear(inter_dim, latent_dim)
        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim,64 ),
            nn.ReLU(),
            nn.Linear(64, 128),  
            nn.ReLU(),
            nn.Linear(128, 256),  
            nn.ReLU(),
            nn.Linear(256, 512),  
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim),  
            nn.Sigmoid(), 
        )



    def KL_loss(self, mu, logvar):
 
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        return kl_loss


    def Recon_loss(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu



    def forward(self,x):
        org_size = x.size()
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)
        return recon_x, mu, logvar

