import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from app.services.gender_classifier import GenderClassifier


class SimpleImageFolder(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform
        classes = {"male": 0, "men": 0, "female": 1, "women": 1}
        for cls_name in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            if cls_name.lower() not in classes:
                continue
            label = classes[cls_name.lower()]
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str, device: str):
    device_t = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SimpleImageFolder(data_dir, transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {data_dir}. Expected subfolders 'male'/'female' or 'men'/'women'.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = GenderClassifier(num_classes=2).to(device_t)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device_t)
            labels = labels.to(device_t)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
    }, out_path)
    print(f"Saved model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train gender classifier (male/female)")
    parser.add_argument("data_dir", help="Path to dataset root containing male/ and female/ folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/gender_classifier.pth")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out, args.device)


if __name__ == "__main__":
    main()



