from PIL import Image
from glob import glob
from NeuralNetwork import NeuralNetwork
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
        bestValAcc = 0
        trainLosses, trainAccuracies = [], []
        valLosses, valAccuracies = [], []

        for epoch in range(self.neuralNetwork.tMax):
            self.neuralNetwork.train()
            print(f"======================Training epoch {epoch}======================")
            runningLoss, correct, total = 0.0, 0, 0

            for inputs, labels in self.neuralNetwork.train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                self.neuralNetwork.optimizer.zero_grad()
                outputs = self.neuralNetwork(inputs)
                
                loss = self.neuralNetwork.loss_function(outputs, labels)
                loss.backward()

                self.neuralNetwork.optimizer.step()

                runningLoss += loss.item() * labels.size(0)
                predicted = outputs.argmax(dim = 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
            trainAcc = 100 * correct / total
            trainAccuracies.append(trainAcc)
            epochLoss = runningLoss / total

            print(f"    Train Loss:             {epochLoss:.4f}")
            print(f"    Train Accuracy:         {trainAcc:.2f}%")

            trainLosses.append(epochLoss)

            valAcc, valLoss = self.Validate()
            valAccuracies.append(valAcc)
            valLosses.append(valLoss)

            if valAcc > bestValAcc:
                bestValAcc = valAcc
                self.SaveModel("best_model.pth")

            self.neuralNetwork.scheduler.step()

        self.SaveModel("trained_net.pth")
        self.ShowPlot(trainLosses, valLosses, trainAccuracies, valAccuracies, True)

    def SaveModel(self, name):
        torch.save(self.neuralNetwork.state_dict(), f"model/{name}")
        print("Model " + name + " saved")

    def ShowPlot(self, data1, data2, data3, data4, save):
        name = f"Loss-Accuracy, lr - {self.neuralNetwork.learningRate}, momentum - {self.neuralNetwork.momentum}, decay - {self.neuralNetwork.decay}, scheduler - True"
        plt.figure(figsize = (10, 4))
        plt.subplot(1, 2, 1)#{
        plt.grid(True)
        plt.plot(data1, label = "Train Loss")
        plt.plot(data2, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")
        plt.legend()
        #}
        plt.subplot(1, 2, 2)#{
        plt.grid(True)
        plt.plot(data3, label = "Train Accuracy")
        plt.plot(data4, label = "Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")
        plt.legend()
        #}

        if(save):
            plt.savefig(f"figure/{name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def ShowConfusionMatrix(self, save):
        self.neuralNetwork.eval()

        numClasses = 10
        confusionMatrix = torch.zeros(numClasses, numClasses, dtype=torch.int64)

        with torch.no_grad():
            for inputs, labels in self.neuralNetwork.test_loader:  # lub test_loader
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.neuralNetwork(inputs)
                _, predicted = torch.max(outputs, 1)

                for trueLabel, predictedLabel in zip(labels.view(-1), predicted.view(-1)):
                    confusionMatrix[trueLabel.long(), predictedLabel.long()] += 1

        confusionMatrix = confusionMatrix.cpu().numpy()

        classNames = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        plt.figure(figsize=(8, 6))
        plt.imshow(confusionMatrix, interpolation="nearest")
        plt.colorbar()

        plt.xticks(range(numClasses), classNames, rotation=45)
        plt.yticks(range(numClasses), classNames)

        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")

        # wpisanie wartości do komórek
        for i in range(numClasses):
            for j in range(numClasses):
                plt.text(
                    j, i,
                    str(confusionMatrix[i, j]),
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        if save:
            name = (
                f"ConfusionMatrix, lr - {self.neuralNetwork.learningRate}, "
                f"momentum - {self.neuralNetwork.momentum}, "
                f"decay - {self.neuralNetwork.decay}"
            )
            plt.savefig(f"figure/{name}.png", dpi=300, bbox_inches="tight")

        plt.show()

    def Validate(self):
        runningLoss, correct, total = 0.0, 0, 0
        self.neuralNetwork.eval()

        with torch.no_grad():
            for images, labels in self.neuralNetwork.val_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                outputs = self.neuralNetwork(images)
                predicted = outputs.argmax(dim = 1)

                loss = self.neuralNetwork.loss_function(outputs, labels)
                runningLoss += loss.item() * labels.size(0)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avgLoss = runningLoss / total

        print(f"    Validation Loss:        {avgLoss:.4f}")
        print(f"    Validation Accuracy:    {accuracy:.2f}%")

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
        self.ShowConfusionMatrix(True)
    
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

    def Experiment(self):
        lrs = [0.1, 0.001, 0.00001]
        lrMins = [0.0001, 0.000001, 0.00000001]
        momentums = [0.9, 0.7, 0.5]
        epochs = [20]

        for epoch in epochs:
            for momentum in momentums:
                for lrMin in lrMins:
                    for lr in lrs:
                        self.neuralNetwork = NeuralNetwork().to(self.device)

                        self.neuralNetwork.learningRate = lr
                        self.neuralNetwork.momentum = momentum
                        self.neuralNetwork.learningRateMin = lrMin
                        self.neuralNetwork.tMax = epoch

                        self.TestExperiment()

    def TestExperiment(self):
        print("Experimenting")
        bestValAcc = 0
        trainAcc = 0
        total = 0
        correct = 0
        trainLosses = []
        trainAccuracies = []
        valLosses = []
        valAccuracies = []
    
        self.neuralNetwork.train()

        for epoch in range(self.neuralNetwork.tMax):
            print(f"======================Training epoch {epoch}======================")
            running_loss = 0.0

            for i, data in enumerate(self.neuralNetwork.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                self.neuralNetwork.optimizer.zero_grad()
                outputs = self.neuralNetwork(inputs)
                _, predicted = torch.max(outputs, 1)

                loss = self.neuralNetwork.loss_function(outputs, labels)
                loss.backward()
                self.neuralNetwork.optimizer.step()

                running_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            trainAcc = 100 * correct / total
            trainAccuracies.append(trainAcc)

            print(f"    Train Loss:             {running_loss / len(self.neuralNetwork.train_loader):.4f}")
            print(f"    Train Accuracy:         {trainAcc:.2f}%")

            trainLosses.append(running_loss / len(self.neuralNetwork.train_loader))
            valAcc, valLoss = self.Validate()
            valAccuracies.append(valAcc)
            valLosses.append(valLoss)
            self.neuralNetwork.train()

            if valAcc > bestValAcc:
                bestValAcc = valAcc
                torch.save(self.neuralNetwork.state_dict(), "model/best_model.pth")
                print("    Saved best model")

            self.neuralNetwork.scheduler.step()

        torch.save(self.neuralNetwork.state_dict(), "model/trained_net.pth")
        print("    Network trained and saved")
        self.ShowPlot(trainLosses, valLosses, trainAccuracies, valAccuracies, True)

                        
