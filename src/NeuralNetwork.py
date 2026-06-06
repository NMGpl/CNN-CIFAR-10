import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.learningRate = 0.001
        self.momentum = 0.9
        self.decay = 1.2e-6
        self.tMax = 30
        self.learningRateMin = 0.000001

        # self.InitLayers()

        self.features = self.initFeatures()
        self.classifier = self.initClassifier()

        self.PrepareTransforms()

        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.parameters(), lr = self.learningRate, momentum = self.momentum, weight_decay = self.decay)
        self.optimizer = optim.AdamW(self.parameters(), lr = self.learningRate, weight_decay = self.decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.tMax, eta_min = self.learningRateMin)
        
    def initFeatures(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding = 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding = 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
    
    def initClassifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def PrepareTransforms(self):
        self.trainTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness = 0.2, 
                contrast = 0.2, 
                saturation = 0.2
                ),
            transforms.RandomGrayscale(0.1),
            transforms.RandomCrop(
                32, 
                padding = 4
                ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        self.testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])