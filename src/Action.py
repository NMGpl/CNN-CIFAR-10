from PIL import Image
import torch
import torchvision.transforms as transforms

class Action:
    def __init__(self, neuralNetwork, device):
        self.device = device
        self.neuralNetwork = neuralNetwork

    def Train(self, trainingLength):
        print("Training network")
        best_acc = 0
        self.neuralNetwork.train()

        for epoch in range(trainingLength):
            print(f"Training epoch {epoch}...")
            running_loss = 0.0

            for i, data in enumerate(self.neuralNetwork.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.neuralNetwork.optimizer.zero_grad()
                outputs = self.neuralNetwork(inputs)
                loss = self.neuralNetwork.loss_function(outputs, labels)
                loss.backward()
                self.neuralNetwork.optimizer.step()
                running_loss += loss.item()
            
            print(f"Loss: {running_loss / len(self.neuralNetwork.train_loader):.4f}")

            val_acc = self.Validate()
            self.neuralNetwork.train()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.neuralNetwork.state_dict(), "model/best_model.pth")
                print("Saved best model")

        torch.save(self.neuralNetwork.state_dict(), "model/trained_net.pth")
        print("Network trained and saved")

    def Validate(self):
        correct = 0
        total = 0
        self.neuralNetwork.eval()

        with torch.no_grad():
            for images, labels in self.neuralNetwork.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.neuralNetwork(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")

        return accuracy

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
                images, labels = images.to(self.device), labels.to(self.device)
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

        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def CustomTest(self):
        print("Custom test")
        image_paths = ["1.jpg", "2.jpg"]
        images = [self.LoadCustomImage(img) for img in image_paths]

        self.neuralNetwork.eval()
        with torch.no_grad():
            for image in images:
                image = image.to(self.device)
                output = self.neuralNetwork(image)
                _, predicted = torch.max(output, 1)
                print(f"Prediction: {self.neuralNetwork.class_names[predicted.item()]}")