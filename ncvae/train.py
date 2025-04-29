#FIXME: CLEAN THIS SHIT
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import *
from model import *

def train_nc_vae(X, vae: NCVAE, discriminator: NCVAEDiscriminator, wandb_project: str = "nce-vae", wandb_name: str = "kanna-kanna-kanna-kanna-kanna-chameleon", device: str = 'cpu'):

    vae.to(device)
    discriminator.to(device)

    # Prepare Dataset
    dataloader_recon, dataloader_nce = generate_datasets(X)

    # WanDB
    wandb.init(project=wandb_project, name=wandb_name)

    optimizer_generative = torch.optim.Adam(vae.parameters(), lr=1e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(1000):
        recon_iter = iter(dataloader_recon)
        nce_iter = iter(dataloader_nce)

        while True:
            try:
                batch_recon = next(recon_iter)[0].to(device)
                batch_nce = next(nce_iter)[0].to(device)
            except StopIteration:
                # If either iterator runs out of data, end epoch
                break

            discriminator.train()
            optimizer_discriminator.zero_grad()

            with torch.no_grad():
                z_true, _ = vae.message_sender.forward(batch_nce)
                z_sampled = vae.message_sender.sample(batch_nce)

            pred_true = discriminator(z_true)
            pred_sampled = discriminator(z_sampled)

            bce_loss_true = F.binary_cross_entropy(pred_true, torch.ones_like(pred_true))
            bce_loss_sampled = F.binary_cross_entropy(pred_sampled, torch.zeros_like(pred_sampled))
            discriminator_loss = bce_loss_true + bce_loss_sampled

            discriminator_loss.backward()
            optimizer_discriminator.step()

            vae.train()
            optimizer_generative.zero_grad()

            # Reconstruction Loss
            reconstruction_loss = vae.loss(batch_recon)

            # Fresh NCE loss (on Set B latents)
            z_true, _ = vae.message_sender.forward(batch_nce)
            z_sampled = vae.message_sender.sample(batch_nce)

            pred_true = discriminator(z_true)
            pred_sampled = discriminator(z_sampled)

            nce_loss = -(torch.log(pred_true + 1e-8).mean() + torch.log(1 - pred_sampled + 1e-8).mean())

            vae_loss = reconstruction_loss + nce_loss

            vae_loss.backward()
            optimizer_generative.step()

            wandb.log({
                'vae_loss': vae_loss.item(),
                'reconstruction_loss': reconstruction_loss.item(),
                'nce_loss': nce_loss.item(),
                'discriminator_loss': discriminator_loss.item(),
            })
        
