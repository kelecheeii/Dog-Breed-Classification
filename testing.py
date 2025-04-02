import torch
from torchvision import transforms
from Dog_Breed import ImageOnlyClassifier  # Import model
from Dog_Breed import test_loader  # Import test_loader

# Define model & load best weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 70  

model = ImageOnlyClassifier(num_classes=num_classes)  # Initialize model
model.load_state_dict(torch.load("/home/kelechi.mbibi/Final_Project/Best_DogBreed_Model.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# Define loss function for evaluation
criterion = torch.nn.CrossEntropyLoss()

# Correct Evaluation Function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy

#Run Evaluation
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}")
