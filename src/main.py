import os
import torch
from NeuralNetwork import NeuralNetwork
from DataLoader import DataLoader
from Action import Action

def printMenu():
    print("1) Train network")
    print("2) Load final network")
    print("3) Load best network")
    print("4) Test network")
    print("5) Custom test")

def main():
    torch.backends.cudnn.benchmark = True
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    neuralNetwork = NeuralNetwork().to(device)
    
    dataLoader = DataLoader(neuralNetwork)
    action = Action(neuralNetwork, device)


    while(True):
        print("Using", device)
        printMenu()
        choice = input()
        choice = int(choice)
        
        if(choice == 1):
            os.system("cls")
            # Train()
            action.Train()
        
        elif(choice == 2):
            os.system("cls")
            # Load("trained_net.pth")
            action.Load("model/trained_net.pth")

        elif(choice == 3):
            os.system("cls")
            # Load("best_model.pth")
            action.Load("model/best_model.pth")

        elif(choice == 4):
            os.system("cls")
            # Test()
            action.Test()

        elif(choice == 5):
            os.system("cls")
            # CustomTest()
            action.CustomTest()

        else:
            os.system("cls")

if __name__ == "__main__":
    main()