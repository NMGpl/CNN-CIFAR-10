from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as img

class Action:
    def __init__(self, neuralNetwork, device):
        self.device = device
        self.neuralNetwork = neuralNetwork

    def Train(self):
        print("Training network")
        best_acc = 0
        trainLosses = []
        valLosses = []
        valAccuracies = []
    
        self.neuralNetwork.train()

        for epoch in range(self.neuralNetwork.tMax):
            print(f"Training epoch {epoch}...")
            running_loss = 0.0

            for i, data in enumerate(self.neuralNetwork.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                self.neuralNetwork.optimizer.zero_grad()
                outputs = self.neuralNetwork(inputs)
                loss = self.neuralNetwork.loss_function(outputs, labels)
                loss.backward()
                self.neuralNetwork.optimizer.step()
                running_loss += loss.item()
            
            print(f"Loss: {running_loss / len(self.neuralNetwork.train_loader):.4f}")

            trainLosses.append(running_loss / len(self.neuralNetwork.train_loader))
            val_acc, valLoss = self.Validate()
            valAccuracies.append(val_acc)
            valLosses.append(valLoss)
            self.neuralNetwork.train()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.neuralNetwork.state_dict(), "model/best_model.pth")
                print("Saved best model")

            self.neuralNetwork.scheduler.step()

        torch.save(self.neuralNetwork.state_dict(), "model/trained_net.pth")
        print("Network trained and saved")
        self.ShowPlot(trainLosses, "Epoch", "Loss", "Training Loss", True)
        self.ShowPlot(valLosses, "Epoch", "Loss", "Validation Loss", True)
        self.ShowPlot(valAccuracies, "Epoch", "Accuracy", "Validation Accuracy", True)

    def ShowPlot(self, data, xlabel, ylabel, title, save):
        name = f"{title}, lr - {self.neuralNetwork.learningRate}, momentum - {self.neuralNetwork.momentum}, decay - {self.neuralNetwork.decay}, scheduler - True"
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(name)
        if(save):
            plt.savefig(f"figure/{name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def Validate(self):
        correct = 0
        total = 0
        running_loss = 0

        self.neuralNetwork.eval()

        with torch.no_grad():
            for images, labels in self.neuralNetwork.val_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                outputs = self.neuralNetwork(images)
                _, predicted = torch.max(outputs, 1)

                loss = self.neuralNetwork.loss_function(outputs, labels)
                running_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avgLoss = running_loss / len(self.neuralNetwork.val_loader)

        print(f"Validation Loss: {avgLoss:.4f}")
        print(f"Accuracy: {accuracy}%")

        return accuracy, avgLoss

    def Load(self, name):
        print("Load network")
        self.neuralNetwork.load_state_dict(torch.load(name, map_location = self.device))
        print("Network " + name + " loaded")

    def Test(self):
        print("Test")
        correct = 0
        total = 0
        self.neuralNetwork.eval()

        with torch.no_grad():
            for data in self.neuralNetwork.test_loader:
                images, labels = data
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.neuralNetwork(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
    
    def LoadCustomImage(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def CustomTest(self):
        print("Custom test")
        image_paths = self.GetImgsPath()
        images = [self.LoadCustomImage(img) for img in image_paths]

        self.neuralNetwork.eval()
        predictions, confidences = self.Predict(images)
        
        for i in range(len(image_paths)):
            image = img.imread(image_paths[i])
            plt.imshow(image)
            print(f"Prediction: {self.neuralNetwork.class_names[predictions[i].item()]}")

            plt.axis("off")
            plt.title(f"Prediction: {self.neuralNetwork.class_names[predictions[i].item()]}, {round(confidences[i].item() * 100, 2)}%")
            plt.pause(1.5)
        
        plt.show()
    
    def GetImgsPath(self):
        path = glob("data/custom/*")
        return path

    def Predict(self, images):
        predictions = []
        confidences = []

        with torch.no_grad():
            for image in images:
                image = image.to(self.device, non_blocking=True)
                output = self.neuralNetwork(image)

                probabilities = F.softmax(output, dim = 1)

                confidence = torch.max(probabilities, 1).values
                _, predicted = torch.max(output, 1)

                confidences.append(confidence)
                predictions.append(predicted)

        return predictions, confidences