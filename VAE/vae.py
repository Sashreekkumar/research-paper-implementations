import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 14 -> 7
        
        self.fc = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 7 -> 14
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 14 -> 28

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = F.relu(self.fc2(x))
        
        x = x.view(-1, 64, 7, 7)
        
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # pixel range [0,1]
        
        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar