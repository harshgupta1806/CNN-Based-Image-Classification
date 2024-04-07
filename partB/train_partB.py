import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import wandb
import random
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
# Import necessary libraries for the project

torch.manual_seed(2)  # Setting the random seed for PyTorch operations to ensure reproducibility
random.seed(2)  # Setting the random seed for Python's built-in random module
np.random.seed(2)  # Setting the random seed for NumPy operations

# Function to determine and set the device for computation (CPU/GPU)
def set_device():
    device = "cpu"  # Defaulting to CPU
    if torch.cuda.is_available():  # Checking if GPU is available
        device = torch.device("cuda")  # Setting device to GPU if available
    else:
        device = torch.device("cpu")  # Otherwise, default to CPU
    return device

device = set_device()  # Calling the function to set the device
print("Currently Using :: ", device)  # Printing the currently used device


import warnings
import argparse

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--datapath', help='give data path e.g. - \'D:/Deep_Learning_A2/nature_12K/inaturalist_12K\'', type=str, default='D:/Deep_Learning_A2/nature_12K/inaturalist_12K')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)
parser.add_argument('-s', '--strategy', help="Choices :- [start, middle, end, freeze_all]", type=str, default="start", choices= ["start", "middle", "end", "freeze_all"])
parser.add_argument('-k', '--layers_to_freeze', help="No. of Layers to freeze", type=int, default=5)
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'adam', choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0001)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.9)
parser.add_argument('-eval', '--evaluate', help='Evaluate model on test data',type=int, default=0, choices=[0, 1])
arguments = parser.parse_args()


def load_data(batch_size,img_size, device):
    train_augmentation = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_augmentation = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Load the dataset
    data_path = arguments.datapath
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform = train_augmentation)
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform = test_augmentation)

    
    labels = train_dataset.classes
    trainset, valset = random_split(train_dataset, [8000, 1999])
    if device == "cuda":
        train_loader = DataLoader(trainset, batch_size = batch_size, num_workers=2)
        val_loader = DataLoader(valset, batch_size = batch_size, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=2)
    else:
        train_loader = DataLoader(trainset, batch_size = batch_size)
        val_loader = DataLoader(valset, batch_size = batch_size)
        test_loader = DataLoader(test_dataset, batch_size = batch_size)

    return labels , train_loader, val_loader, test_loader

def freeze_layers(model, options, k):
    """
    Freeze specified layers of a neural network model.

    Args:
    - model (torch.nn.Module): The neural network model.
    - options (str): Specifies which layers to freeze. Options: "start", "middle", or "end".
    - k (int): Number of layers to freeze. For "start" and "end" options, k specifies the number of layers from the start or end respectively.

    Raises:
    - ValueError: If k is not within the valid range.

    Returns:
    - None
    """

    # Check if k is within the valid range
    if k < 0 or k >= len(list(model.named_children())):
        raise ValueError(f"Invalid value of k. Choose between 0 and {len(list(model.named_children())) - 1}")


# Freeze layers based on the specified option

    # Freeze first k layers
    if options == "start":
        for layer_num, (name, layer) in enumerate(model.named_children(), 1):
            if layer_num <= k:
                for p_name, param in layer.named_parameters():
                    param.requires_grad = False
        print(f"Freezed First {k} Layer")


    # Freeze Middle layers
    elif options == "middle":
        total_layer = len(list(model.named_children()))
        middle_layer = total_layer // 2  # Get the index of the middle layer
        num_layers_to_freeze = k  # Number of layers to freeze around the middle layer

        for layer_num, (name, layer) in enumerate(model.named_children(), 1):
            if middle_layer - num_layers_to_freeze <= layer_num < middle_layer + num_layers_to_freeze:
                for p_name, param in layer.named_parameters():
                    param.requires_grad = False

        start_layer = middle_layer - num_layers_to_freeze
        end_layer = middle_layer + num_layers_to_freeze
        print(f"Freeze middle layers from layer {start_layer} to {end_layer} and Train rest of the layers")


    # Freeze last k layers 
    elif options == "end":
        total_layers = len(list(model.named_children()))
        start_layer = total_layers - k
        end_layer = total_layers
        
        for layer_num, (name, layer) in enumerate(model.named_children(), 1):
            if start_layer <= layer_num <= end_layer:
                for p_name, param in layer.named_parameters():
                    param.requires_grad = False
        
        print(f"Freeze last {k} layers and Train rest of the layers")


    # Freeze all layers (train only last layer)
    elif options == "freeze_all":
        total_layers = len(list(model.named_children()))
        curr_layers = 0
        for name, layer in model.named_children():
            if curr_layers < total_layers - 1:
                for p_name, param in layer.named_parameters():
                    # print(p_name)
                    param.requires_grad = False
            curr_layers += 1

        print(f"Train only last layer and freeze all other layers")
            


def set_optimizer(model, opt_name, PARAM):
    if opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr = PARAM["eta"], momentum=PARAM["momentum"], weight_decay=0)
    elif opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr = PARAM["eta"], weight_decay=0)
    elif opt_name == "nadam":
        opt = optim.NAdam(model.parameters(), lr = PARAM["eta"], weight_decay=0)
    elif opt_name == "adagrad":
        opt = optim.Adagrad(model.parameters(), lr = PARAM["eta"], weight_decay=0)
    elif opt_name == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr = PARAM["eta"], weight_decay=0)
    return opt


def train_model(model, device, PARAM):
    # Freeze the part of Model, based on stretegy
    freeze_layers(model, PARAM["strategy"], PARAM["k"])

    # Set Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = set_optimizer(model, PARAM["optimizer"], PARAM)

    labels, train_loader, val_loader, test_loader = load_data(PARAM["batch_size"], PARAM["image_size"], device)
    for epoch in range(PARAM["epochs"]):
        model.train()
#         print(type(model))

        running_loss = 0.0
        correct = 0
        total = 0
        count = 0
        for images, labels in train_loader:
            # print("image is loaded")
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
#             print(type(output))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(outputs)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            # print(_, predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1
            if count%5 == 0 :
                print(count, end = " | ")
        print("")
        model.eval()
        running_val_loss = 0.0
        correct_pred = 0
        total_pred = 0
    

        with torch.no_grad():
            for val_img, val_label in val_loader:
                val_img = val_img.to(device)
                val_label = val_label.to(device)
                val_output = model(val_img)
                loss_val = criterion(val_output, val_label)
                running_val_loss += loss_val.item()
#                 print(running_val_loss)
                idx, class_ = torch.max(val_output.data, 1)
                total_pred += val_label.size(0)
                correct_pred += (class_ == val_label).sum().item()

        
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Training_Accuracy: {100 * correct / total}%, Validation_Loss : {running_val_loss/len(val_loader)}, Validation_accuracy : {100 * correct_pred / total_pred}%")     
    return model
        
def load_model():
    model = models.googlenet(pretrained=True)

    last_layer_in_features = model.fc.in_features
    model.fc = nn.Linear(last_layer_in_features, 10)
    model = model.to(device)
    return model

## Function to Evaluate Model
def calculate_accuracy_on_test_data(model, device):
    labels, train_loader, val_loader, test_loader = load_data(1, 224, "No", device)
    model.eval()  # Set model to evaluation mode
    running_test_loss = 0.0  # Initialize running loss for validation
    correct_pred = 0  # Initialize correct predictions counter for validation
    total_pred = 0  # Initialize total samples counter for validation
    criterion = nn.CrossEntropyLoss()
    # Evaluate on validation set
    with torch.no_grad():
        for test_img, test_label in test_loader:
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            test_output = model(test_img)
            loss_test = criterion(test_output, test_label)
            running_test_loss += loss_test.item()
            _, class_ = torch.max(test_output.data, 1)
            total_pred += test_label.size(0)
            correct_pred += (class_ == test_label).sum().item()
    print(f"Test Accuracy : {100 * correct_pred / total_pred}%, Test Loss : {running_test_loss/len(test_loader)}")
    return 100 * correct_pred / total_pred, running_test_loss/len(test_loader)

PARAM = {
    "image_size" : 224,
    "batch_size" : arguments.batch_size,
    "eta" : arguments.learning_rate,
    "epochs" : arguments.epochs,
    "output_size" : 10,
    "optimizer" : arguments.optimizer,
    "weight_decay" : 0,
    "strategy" : arguments.strategy,
    "k" : arguments.layers_to_freeze,
    "momentum" : arguments.momentum
}

model = load_model()
model = train_model(model, device, PARAM)
if arguments.evaluate == 1:
    calculate_accuracy_on_test_data(model, device)

