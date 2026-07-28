import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GTSRBDataset
from model import TrafficSignCNN


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(BASE_DIR, "dataset", "GTSRB", "Train.csv")
root_path = os.path.join(BASE_DIR, "dataset", "GTSRB")

dataset = GTSRBDataset(csv_path, root_path)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = TrafficSignCNN()

criterion = nn.CrossEntropyLoss()      # loss function

optimizer = optim.Adam(
    model.parameters(),         # update the CNNs learned weights during training
    lr=0.001                    # learning rate, controls how large each weight update is
)

epochs = 10

best_accuracy = 0

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
            os.path.join(BASE_DIR, "models", "best_model.pth")
    )
        # state_dict() is all of model learned weights and torch.save writes it to the file 

    print(f" Model saved. Accuracy: {accuracy:.2f}%")


"""
Epoch 1/3 | Loss: 1.0614 | Accuracy: 69.65%
Best model saved!
Epoch 2/3 | Loss: 0.1650 | Accuracy: 95.65%
Best model saved!
Epoch 3/3 | Loss: 0.0843 | Accuracy: 97.74%
Best model saved!  1st time testing
"""