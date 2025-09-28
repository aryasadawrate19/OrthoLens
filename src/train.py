# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataloader import create_dataloader
from src.model import get_model

# ==========================
# Config
# ==========================
BATCH_SIZE = 8  # start small for debugging
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_WORKERS = 0  # safe for Windows; set >0 on Linux for speed

# ==========================
# Training + Validation loops
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels, _) in enumerate(loader, start=1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log every 50–100 batches
        if batch_idx % 50 == 0:
            print(f"   [Epoch {epoch}] Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    study_preds = {}
    study_labels = {}

    with torch.no_grad():
        for images, labels, study_ids in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            # Aggregate predictions per study
            for study_id, pred, label in zip(study_ids, preds.cpu().numpy(), labels.cpu().numpy()):
                if study_id not in study_preds:
                    study_preds[study_id] = []
                    study_labels[study_id] = label
                study_preds[study_id].append(pred)

    # Compute study-level accuracy using majority vote
    correct, total = 0, 0
    for study_id in study_preds:
        votes = study_preds[study_id]
        final_pred = max(set(votes), key=votes.count)
        correct += int(final_pred == study_labels[study_id])
        total += 1

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    print(f"Validation [Epoch {epoch}] Loss: {avg_loss:.4f}, Study-level Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


# ==========================
# Main training script
# ==========================
def main():
    print(f"Using device: {DEVICE}")

    print("Loading data...")
    train_loader = create_dataloader(
        "MURA-v1.1/train_image_paths.csv",
        "MURA-v1.1/train_labeled_studies.csv",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = create_dataloader(
        "MURA-v1.1/valid_image_paths.csv",
        "MURA-v1.1/valid_labeled_studies.csv",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f"Data loaded: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    print("Initializing model...")
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("Model ready.")

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        print(f"Avg Train Loss: {train_loss:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, epoch)

        # Save checkpoint
        model_path = OUTPUT_DIR / f"model_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved checkpoint: {model_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = OUTPUT_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (acc={best_val_acc:.4f})")

    print("Training complete!")


if __name__ == "__main__":
    main()
