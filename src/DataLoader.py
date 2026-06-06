import torch
import torchvision

class DataLoader:
    def __init__(self, neuralNetwork, device):
        self.device = device
        self.BATCH_SIZE = 256
        self.NUM_WORKERS = 6
        self.PrepareDataset(neuralNetwork)
        self.LoadDataset(neuralNetwork)

    def PrepareDataset(self, neuralNetwork):
        neuralNetwork.train_data = torchvision.datasets.CIFAR10(root = "./data", train = True, transform = neuralNetwork.trainTransform, download = True)
        neuralNetwork.test_data = torchvision.datasets.CIFAR10(root = "./data", train = False, transform = neuralNetwork.testTransform, download = True)
        neuralNetwork.train_size = int(0.8 * len(neuralNetwork.train_data))
        neuralNetwork.val_size = len(neuralNetwork.train_data) - neuralNetwork.train_size

        neuralNetwork.train_dataset, neuralNetwork.val_dataset = torch.utils.data.random_split(
            neuralNetwork.train_data, [neuralNetwork.train_size, neuralNetwork.val_size]
        )

    def LoadDataset(self, neuralNetwork):
        neuralNetwork.train_loader = torch.utils.data.DataLoader(neuralNetwork.train_dataset, batch_size = self.BATCH_SIZE, shuffle=True, num_workers = self.NUM_WORKERS, pin_memory = True, persistent_workers = True)
        neuralNetwork.val_loader = torch.utils.data.DataLoader(neuralNetwork.val_dataset, batch_size = self.BATCH_SIZE, shuffle=False, num_workers = self.NUM_WORKERS, pin_memory = True, persistent_workers = True)
        neuralNetwork.test_loader = torch.utils.data.DataLoader(neuralNetwork.test_data, batch_size = self.BATCH_SIZE, shuffle = False, num_workers = self.NUM_WORKERS, pin_memory = True, persistent_workers = True)

        neuralNetwork.image, neuralNetwork.label = neuralNetwork.train_data[0]
        neuralNetwork.image.size()
        neuralNetwork.class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]