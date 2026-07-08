import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device('cpu')

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 3
latent_dim = 20

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Subsample for very fast portfolio generation
train_dataset.data = train_dataset.data[:2000]
train_dataset.targets = train_dataset.targets[:2000]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim) # mu
        self.fc22 = nn.Linear(400, latent_dim) # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []

print("Starting VAE training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')

os.makedirs('website/assets/vae', exist_ok=True)

# Plot Loss
plt.figure(figsize=(6,4))
plt.plot(losses, marker='o', color='purple')
plt.title('VAE Training Loss (Reconstruction + KLD)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('website/assets/vae/loss_plot.png')

# Generate samples
model.eval()
with torch.no_grad():
    sample = torch.randn(16, latent_dim).to(device)
    sample = model.decode(sample).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(4,4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sample[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('website/assets/vae/generated_samples.png')

print("VAE training complete.")
