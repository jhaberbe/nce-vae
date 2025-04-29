import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def generate_datasets(X: torch.Tensor, X_batch: torch.Tensor, batch_size: int = 512):
    dataset = TensorDataset(X, X_batch)
    dataset_recon, dataset_nce = random_split(dataset, [0.5, 0.5])
    dataloader_recon = DataLoader(dataset_recon, batch_size=batch_size, shuffle=True)
    dataloader_nce = DataLoader(dataset_nce, batch_size=batch_size, shuffle=True)
    return dataloader_recon, dataloader_nce
