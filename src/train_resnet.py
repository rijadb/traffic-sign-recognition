import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_resnet import GTSRBDataset
from resnet_model import TrafficSignResNet


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(BASE_DIR, "dataset", "GTSRB", "Train.csv")
root_path = os.path.join(BASE_DIR, "dataset", "GTSRB")

dataset = GTSRBDataset(csv_path, root_path)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = TrafficSignResNet(pretrained=True)

for param in model.model.parameters():
    param.requires_grad = False

# Train the last residual block
for param in model.model.layer4.parameters():
    param.requires_grad = True

# Train the classifier
for param in model.model.fc.parameters():
    param.requires_grad = True             

criterion = nn.CrossEntropyLoss()      # loss function

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),         # only classifiers parameters
    lr=0.0001                    # learning rate, controls how large each weight update is
)

epochs = 10

best_accuracy = 0

model.train()

for epoch in range(epochs):

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()               # clearing everything form prev. batch

        outputs = model(images)

        loss = criterion(outputs, labels)   # loss/wrong prediction

        loss.backward()                     # going backwards too see which weight contributed to mistake

        optimizer.step()                    # chaning the weights from upper line

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        running_loss += loss.item()         # combining loss values from batches

        accuracy = 100 * correct / total

    print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Accuracy: {accuracy:.2f}%"
    )
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            os.path.join(BASE_DIR, "models", "best_resnet18.pth")
                )
        # state_dict() is all of model learned weights and torch.save writes it to the file 

        print(f" Model saved. Accuracy: {accuracy:.2f}%")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")