import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from vae import VAE

# Data

transform = transforms.ToTensor()

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Loss
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (pixel-wise)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(latent_dim=20).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)

        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")