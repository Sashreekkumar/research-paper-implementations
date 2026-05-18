import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae import VAE


def vae_loss(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

train_total, train_recon, train_kl = [], [], []

for epoch in range(epochs):
    model.train()
    total_epoch_loss = 0

    for x, _ in loader:
        x = x.to(device)

        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total.append(loss.item() / x.size(0))
        train_recon.append(recon_l.item() / x.size(0))
        train_kl.append(kl_l.item() / x.size(0))

        total_epoch_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_epoch_loss / len(dataset):.4f}")
