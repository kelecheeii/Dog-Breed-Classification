
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision import transforms, models
import numpy as np
import os
import torch.optim as optim

# Checking for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define dataset paths
train_dir = '/home/kelechi.mbibi/dog_breed/train'
val_dir = '/home/kelechi.mbibi/dog_breed/valid'
test_dir = '/home/kelechi.mbibi/dog_breed/test'

# Count images recursively in all subfolders
train_files = glob.glob(os.path.join(train_dir, '**', '*.*'), recursive=True)
val_files = glob.glob(os.path.join(val_dir, '**', '*.*'), recursive=True)
test_files = glob.glob(os.path.join(test_dir, '**', '*.*'), recursive=True)

# Print counts
print(f" Train images: {len(train_files)}")
print(f" Validation images: {len(val_files)}")
print(f" Test images: {len(test_files)}")

# Data augmentation
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(root=train_dir, transform=transform_train)
val_dataset = ImageFolder(root=val_dir, transform=transform_test)
test_dataset = ImageFolder(root=test_dir, transform=transform_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Image-only Classifier
class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageOnlyClassifier, self).__init__()
        self.image_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.image_model(x)


# Training Setup
num_classes = 70
model = ImageOnlyClassifier(num_classes=num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

# Early Stopping Variables
best_loss = float("inf")
patience = 3
counter = 0

if __name__ == "__main__" and not getattr(__builtins__, "__TESTING__", False):
    
    for epoch in range(30):
        print(f"\nEpoch {epoch+1}/30")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "Best_DogBreed_Model.pth")
            print("Model saved!")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()
