# src/inference_visual.py
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import get_model
from src.dataset import MURADataset
from collections import defaultdict

# ==========================
# Config
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_SAMPLES = 8  # number of validation images to display
MODEL_PATH = "outputs/models/best_model.pth"

# ==========================
# Transforms (same as validation)
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Inverse transform to display images correctly
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    transforms.ToPILImage()
])

# ==========================
# Load model
# ==========================
model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ==========================
# Load validation dataset
# ==========================
dataset = MURADataset(
    "MURA-v1.1/valid_image_paths.csv",
    "MURA-v1.1/valid_labeled_studies.csv",
    transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ==========================
# Run inference and display
# ==========================
study_preds = defaultdict(list)
study_labels = {}
num_images = 0

with torch.no_grad():
    for images, labels, study_ids in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for i, (img_tensor, pred, label, study_id) in enumerate(zip(images, preds.cpu().numpy(), labels.cpu().numpy(), study_ids)):
            study_preds[study_id].append(pred)
            study_labels[study_id] = label

            # Display image
            img_display = inv_transform(img_tensor.cpu())
            final_pred = max(set(study_preds[study_id]), key=study_preds[study_id].count)
            correct_text = "Correct" if final_pred == study_labels[study_id] else "Incorrect"

            plt.imshow(img_display)
            plt.title(f"Study: {study_id}\nTrue: {label} | Predicted: {final_pred} | {correct_text}")
            plt.axis('off')
            plt.show()

        num_images += images.size(0)
        if num_images >= NUM_SAMPLES:
            break
