import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root = "./data", train = True, transform = transform, download = True)
test_data = torchvision.datasets.CIFAR10(root = "./data", train = False, transform = transform, download = True)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_data, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 128, shuffle = False, num_workers = 2)

image, label = train_data[0]
image.size()
class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Using device:", device)

net = NeuralNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
# optimizer = optim.Adam(net.parameters(), lr = 0.001)

def Train():
    best_acc = 0
    net.train()

    for epoch in range(75):
        print(f"Training epoch {epoch}...")
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Loss: {running_loss / len(train_loader):.4f}")

        val_acc = Validate()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), "best_model.pth")
            print("Saved best model")

    torch.save(net.state_dict(), "trained_net.pth")
    print("Network trained and saved")

def Load(name):
    global net
    net = NeuralNet().to(device)
    net.load_state_dict(torch.load(name, map_location = device))
    print("Network " + name + " loaded")

def main():
    while(True):
        print("1) Train network")
        print("2) Load final network")
        print("3) Load best network")
        print("4) Test network")
        print("5) Custom test")
        print("0) Exit")
        choice = input()
        choice = int(choice)

        if(choice == 1):
            os.system("cls")
            Train()
        
        elif(choice == 2):
            os.system("cls")
            Load("trained_net.pth")

        elif(choice == 3):
            os.system("cls")
            Load("best_model.pth")

        elif(choice == 4):
            os.system("cls")
            Test()

        elif(choice == 5):
            os.system("cls")
            CustomTest()

        elif(choice == 0):
            break
        
        else:
            os.system("cls")

def Validate():
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

    return accuracy

def Test():
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

    return accuracy

def LoadImage(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def CustomTest():
    image_paths = ["1.jpg", "2.jpg"]
    images = [LoadImage(img) for img in image_paths]

    net.eval()
    with torch.no_grad():
        for image in images:
            image = image.to(device)
            output = net(image)
            _, predicted = torch.max(output, 1)
            print(f"Prediction: {class_names[predicted.item()]}")

if __name__ == "__main__":
    main()