import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
import wandb

from .data import *
from .model import *

def train_nc_vae(
        X, 
        X_batch,
        vae, 
        discriminator, 
        wandb_project="nce-vae-batched",
        wandb_name="kanna-kanna-kanna-kanna-kanna-chameleon"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    vae.to(device)
    discriminator.to(device)

    dataloader_recon, dataloader_nce = generate_datasets(X, X_batch)

    wandb.init(project=wandb_project, name=wandb_name)

    optimizer_generative = torch.optim.Adam(vae.parameters(), lr=1e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(1000):
        recon_iter = iter(dataloader_recon)
        nce_iter = iter(dataloader_nce)

        while True:
            try:
                batch_recon_x, batch_recon_batch = next(recon_iter)
                batch_nce_x, batch_nce_batch = next(nce_iter)
                batch_recon_x = batch_recon_x.to(device)
                batch_recon_batch = batch_recon_batch.to(device)
                batch_nce_x = batch_nce_x.to(device)
                batch_nce_batch = batch_nce_batch.to(device)
            except StopIteration:
                break

            # --- Train Discriminator ---
            discriminator.train()
            optimizer_discriminator.zero_grad()

            with torch.no_grad():
                z_true, _ = vae.message_sender.forward(batch_nce_x)
                z_sampled = vae.message_sender.sample(batch_nce_x)

            pred_true = discriminator(z_true)
            pred_sampled = discriminator(z_sampled)

            bce_loss_true = F.binary_cross_entropy(pred_true, torch.ones_like(pred_true))
            bce_loss_sampled = F.binary_cross_entropy(pred_sampled, torch.zeros_like(pred_sampled))
            discriminator_loss = bce_loss_true + bce_loss_sampled

            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_discriminator.step()

            # --- Train VAE ---
            vae.train()
            optimizer_generative.zero_grad()

            # Reconstruction Loss
            reconstruction_loss = vae.loss(batch_recon_x, batch_recon_batch)

            # Fresh NCE loss
            z_true, _ = vae.message_sender.forward(batch_nce_x)
            z_sampled = vae.message_sender.sample(batch_nce_x)
            pred_true = discriminator(z_true)
            pred_sampled = discriminator(z_sampled)
            nce_loss = -(torch.log(pred_true + 1e-8).mean() + torch.log(1 - pred_sampled + 1e-8).mean())

            vae_loss = reconstruction_loss + nce_loss

            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_generative.step()

            wandb.log({
                'vae_loss': vae_loss.item(),
                'reconstruction_loss': reconstruction_loss.item(),
                'nce_loss': nce_loss.item(),
                'discriminator_loss': discriminator_loss.item(),
            })
