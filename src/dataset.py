# src/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms

class MURADataset(Dataset):
    """
    PyTorch Dataset for the MURA dataset.
    Returns:
        image (Tensor): preprocessed image
        label (Tensor): 0 (normal) or 1 (abnormal)
        study_id (str): unique identifier for study-level aggregation
    """

    def __init__(self, image_csv, labeled_csv, transform=None):
        """
        Args:
            image_csv (str): CSV file containing image paths (one per line)
            labeled_csv (str): CSV file containing study folder paths + labels
            transform (callable, optional): torchvision transforms to apply
        """
        # Load image paths
        self.images = pd.read_csv(image_csv, header=None)[0].tolist()

        # Load study-level labels
        labeled_df = pd.read_csv(labeled_csv, header=None)
        labeled_df.columns = ["study_path", "label"]

        # Normalize all study paths and store in dictionary
        self.study_label_dict = {}
        for path, label in zip(labeled_df["study_path"], labeled_df["label"]):
            norm_path = str(Path(path)).replace("\\", "/").rstrip("/") + "/"
            self.study_label_dict[norm_path] = int(label)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.images[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Extract study path as unique ID
        study_path = str(Path(img_path).parent).replace("\\", "/").rstrip("/") + "/"

        # Retrieve label
        label = self.study_label_dict.get(study_path, -1)
        if label == -1:
            raise ValueError(f"Label not found for image {img_path} (study path: {study_path})")

        return image, torch.tensor(label, dtype=torch.long), study_path


# ----------------------------
# Optional utility functions
# ----------------------------
def get_transforms(mode="train"):
    """
    Returns torchvision transforms for training or validation.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def load_dataset(image_csv, label_csv, mode="train"):
    """
    Utility function to create a MURA dataset with proper transforms.
    """
    transform = get_transforms(mode)
    return MURADataset(image_csv, label_csv, transform=transform)


# ----------------------------
# Quick test block
# ----------------------------
if __name__ == "__main__":
    # Example usage
    train_dataset = load_dataset(
        "data/train_image_paths.csv",
        "data/train_labeled_studies.csv",
        mode="train"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0  # use 0 for Windows
    )

    # Test first batch
    images, labels, study_ids = next(iter(train_loader))
    print("Images batch shape:", images.shape)
    print("Labels batch:", labels)
    print("Study IDs batch:", study_ids)
