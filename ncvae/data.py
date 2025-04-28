import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def generate_datasets(X: torch.tensor, batch_size: int = 512):
    # Split
    dataset = TensorDataset(X)
    dataset_recon, dataset_nce = random_split(dataset, [.5, .5])

    # Generate Data Loaders
    dataloader_recon = DataLoader(dataset_recon, batch_size=batch_size, shuffle=True)
    dataloader_nce = DataLoader(dataset_nce, batch_size=batch_size, shuffle=True)
    
    return dataloader_recon, dataloader_nce

