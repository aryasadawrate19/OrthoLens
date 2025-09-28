# src/dataloader.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.dataset import MURADataset

def create_dataloader(image_csv, labeled_csv, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the MURA dataset.

    Args:
        image_csv (str): Path to CSV containing image file paths.
        labeled_csv (str): Path to CSV containing study-level labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for data loading.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = MURADataset(image_csv=image_csv, labeled_csv=labeled_csv, transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader
