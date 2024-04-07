# import libraries
import torch.nn as nn  # Importing neural network module from PyTorch
import torch.optim as optim  # Importing optimization module from PyTorch
from torch.nn import functional as F  # Importing functional API from PyTorch for neural network operations
from torch.utils.data import DataLoader, random_split  # Importing data loading utilities from PyTorch
import pytorch_lightning as pl  # Importing PyTorch Lightning for simplified training
import torchvision.transforms as transforms  # Importing transformation utilities from torchvision
import torchvision.datasets as datasets  # Importing standard datasets from torchvision
import os  # Importing operating system related functionalities
import torch  # Importing PyTorch library
import wandb  # Importing Weights & Biases library for experiment tracking
import random  # Importing random number generation utilities
import numpy as np  # Importing NumPy library for numerical operations
import matplotlib.pyplot as plt # Importing pyplot for ploting grid of images 
import warnings
import argparse


# set seed
torch.manual_seed(2)  # Setting the random seed for PyTorch operations to ensure reproducibility
random.seed(2)  # Setting the random seed for Python's built-in random module
np.random.seed(2)  # Setting the random seed for NumPy operations

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment2')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m026')
parser.add_argument('-d', '--datapath', help='give data path e.g. - \'D:/Deep_Learning_A2/nature_12K/inaturalist_12K\'', type=str, default='D:/Deep_Learning_A2/nature_12K/inaturalist_12K')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)
parser.add_argument('-org', '--filter_org', help="Defines the organization of filter within Network", type=str, default="same", choices= ["same", "double", "half", "alternating_list", "d_alternating_list", "desc", "asc"])
parser.add_argument('-f_s', '--filter_size', help="input list for filter size/kernel size for each layer e.g 11, 9, 7, 5, 3", type=int,  nargs='+', default=[3, 3, 3, 3, 3])
parser.add_argument('-f_n', '--filter_num', help="number of filter in first layer", type=int, default=64)
parser.add_argument('-pfs', '--pool_filter_size', help="pool filter size", type=int, default=2)
parser.add_argument('-dp', '--dropout', help="dropout in last layer", type=float, default=0.3)
parser.add_argument('-aug', '--augmentation', help="Choices :: [Yes, No]", type=str, default="No", choices=["Yes", "No"])
parser.add_argument('-norm', '--batch_norm', help="Choices :: [Yes, No]", type=str, default="Yes", choices= ["Yes", "No"])
parser.add_argument('-img', '--image_size', help="Image Size e.g. 256 (Image Size = 256 X 256)", type=int, default=224)
parser.add_argument('-c_p', '--conv_padding', help="Convolution layer padding", type=int, default=1)
parser.add_argument('-c_s', '--conv_stride', help="Convolution Layer Stride", type=int, default=1)
parser.add_argument('-p_p', '--pool_padding', help="Pooling Layer padding", type=int, default=0)
parser.add_argument('-p_s', '--pool_stride', help="Pooling Layer stride", type=int, default=2)
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'adam', choices= ["sgd", "rmsprop", "adam", "nadam", "adagrad"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0001)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.9)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', type=str, default='random')
parser.add_argument('-ndl', '--neurons_fc', help='Number of neurons in dense layer',type=int, default=256)
parser.add_argument('-a', '--activation', help='choices: ["ReLU", "LeakyReLU", "GELU", "SiLU", "Mish"]', type=str, default='ReLU', choices=["ReLU", "LeakyReLU", "GELU", "SiLU", "Mish"])
parser.add_argument('-p', '--console', help='print training_accuracy + loss, validation_accuracy + loss for every epochs', choices=[0, 1], type=int, default=1)
parser.add_argument('-wl', '--wandb_log', help='log on wandb', choices=[0, 1], type=int, default=0)
parser.add_argument('-plt', '--plot_grid', help='plot grid of 10 X 3 of random images from test data', choices=[0, 1], type=int, default=0)
parser.add_argument('-eval', '--evaluate', help='get test accuarcy and test loss', choices=[0, 1], type=int, default=0)
arguments = parser.parse_args()


#wandb key
# if arguments.wandb_log == 1:
#     wandb.login(key = '57566fbb0e091de2e298a4320d872f9a2b200d12')

def load_data(batch_size, img_size, augmentation = "No", device = "cpu"):
    """
    Function to appropriately load data.

    Args:
    - batch_size (int): Batch size for data loaders.
    - img_size (int): Size of the images after resizing.
    - augmentation (str): Whether to apply data augmentation ("Yes" or "No").

    Returns:
    - labels (list): List of class labels.
    - train_loader (DataLoader): DataLoader for training dataset.
    - val_loader (DataLoader): DataLoader for validation dataset.
    - test_loader (DataLoader): DataLoader for test dataset.
    """

    # Define transformations based on augmentation choice
    if augmentation == "Yes":
        train_augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
        ])
    elif augmentation == "No":
        train_augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    test_augmentation = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    data_path = arguments.datapath  # Path to the dataset directory
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_augmentation)
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=test_augmentation)

    # Splitting the train dataset into train and validation sets
    split_ratio = 0.2  # Ratio of validation set to train set
    len_train_data = int(np.ceil(len(train_dataset) * (1 - split_ratio)))  # Length of training data
    len_val_data = int(len(train_dataset) - len_train_data)  # Length of validation data
    labels = train_dataset.classes  # Extracting class labels

    trainset, valset = random_split(train_dataset, [len_train_data, len_val_data])  # Splitting the dataset

    # Creating DataLoaders for train, validation, and test sets
    if device == "cuda" :
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, num_workers=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return labels, train_loader, val_loader, test_loader


# ConvolutionalNeuralNetwork Class
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, PARAM) -> None:
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten layer to convert 2D input to 1D
        self.filter_org = PARAM["filter_org"]  # Organization pattern for number of filters
        self.filter_num = PARAM["filter_num"]  # Number of initial filters
        self.activation = PARAM["activation"]  # Activation function for layers
        self.con_layers = PARAM["con_layers"]  # Number of convolutional layers (default :: 5)
        self.den_layers = PARAM["dense_layers"]  # Number of dense layers 
        self.input_channel = PARAM["input_channel"]  # Number of input channels
        self.filter_num_list = self.organize_filters(self.filter_org, self.filter_num, self.con_layers)  # List of filter numbers for each layer
        self.filter_size_list = PARAM["filter_size"]  # List of filter sizes for each layer
        self.act = self.activation_fun(PARAM["activation"])  # Activation function
        self.output_act = self.activation_fun(PARAM["output_activation"])  # Activation function for output layer
        self.padding = PARAM["padding"]  # Padding size for convolutional layers
        self.stride = PARAM["stride"]  # Stride size for convolutional layers
        self.pool_padding = PARAM["pool_padding"]  # Padding size for pooling layers
        self.pool_stride = PARAM["pool_stride"]  # Stride size for pooling layers
        self.dense_output_list = PARAM["dense_output_list"]  # List of output sizes for dense layers
        self.image_size = PARAM["image_size"]  # Input image size
        self.pool_filter_size = PARAM["pool_filter_size"]  # Filter size for pooling layers
        self.dropout_list = PARAM["dropout"]  # List of dropout probabilities for layers
        self.batch_norm = PARAM["batch_norm"]  # Whether to use batch normalization (Choices : Yes / No)
        self.initialize = PARAM["init"]  # Weight initialization method (Choices : random/Xavior)

        # Creating the structure of CNN 
        self.create_con_layers(self.input_channel, self.filter_size_list, self.dense_output_list, self.filter_num_list, self.act, self.pool_filter_size, self.output_act, self.image_size, self.dropout_list)


    def create_con_layers(self, input_channel, filter_size_list, dense_output_list, filter_num_list, act, pool_filter_size, output_act, image_size, dropout_list):
        """
        Method to create convolutional and dense layers.

        Args:
        - input_channel (int): Number of input channels.
        - filter_size_list (list): List of filter sizes for convolutional layers.
        - dense_output_list (list): List of output sizes for dense layers.
        - filter_num_list (list): List of numbers of filters for convolutional layers.
        - act (nn.Module): Activation function for layers.
        - pool_filter_size (int): Filter size for pooling layers.
        - output_act (nn.Module): Activation function for output layer.
        - image_size (int): Input image size.
        - dropout_list (list): List of dropout probabilities for layers.
        """
        self.layers = nn.ModuleList()  # List to store layers
        computations = 0 # Counter for the Computations

        # Convolutional layers
        for i in range(1, self.con_layers+1):
            # Creating convolutional layer followed by activation function and max pooling
            layer = nn.Sequential(nn.Conv2d(input_channel, filter_num_list[i-1], filter_size_list[i-1], padding=self.padding, stride=self.stride, bias=False), act, nn.MaxPool2d(pool_filter_size, padding=self.pool_padding, stride=self.pool_stride))
            
            # Updating image size after convolution and pooling
            image_size = (image_size - filter_size_list[i-1] + 2 * self.padding) // self.stride + 1

            # Add Computations for the current Layer
            computations = computations + ((filter_size_list[i-1] ** 2) * input_channel * (image_size ** 2) * filter_num_list[i-1] + filter_num_list[i-1])

            #update image size after pooling layer
            image_size = (image_size + 2 * self.pool_padding - (1 * (pool_filter_size - 1)) - 1) // self.pool_stride + 1

            input_channel = filter_num_list[i-1]  # Updating input channel for next layer
            self.layers.append(layer)  # Adding layer to the list

        dense_input = filter_num_list[self.con_layers-1] * (image_size ** 2)  # Calculating input size for dense layers
        
        # Dense layers
        for i in range(1, self.den_layers+1):
            # Creating dense layer followed by convolutional layer
            layer = nn.Sequential(nn.Linear(dense_input, dense_output_list[i-1]), act)

            # Add Computations for the current Layer
            computations = computations + (dense_input + 1) * dense_output_list[i-1]

            dense_input = dense_output_list[i-1]  # Updating input size for next layer
            self.layers.append(layer)  # Adding layer to the list
            
        # Output layer
        layer = nn.Sequential(nn.Linear(dense_input, 10), nn.Softmax(dim=1))

        # Add Computations for the current Layer
        computations = computations + (dense_input + 1) * 10

        self.layers.append(layer)  # Adding output layer to the list

        print("Computation :: ", computations)  # Printing total computation required

        # Initialization, batch normalization, and dropout
        for layer in range(self.con_layers + self.den_layers + 1):
            if self.initialize == "Xavier" and isinstance(self.layers[layer], nn.Conv2d):
                nn.init.xavier_uniform_(self.layers[layer].weight)
            if self.batch_norm == "Yes" and layer < self.con_layers:
                self.layers[layer].insert(1, nn.BatchNorm2d(filter_num_list[layer]))
            if (layer < self.con_layers + self.den_layers) and self.dropout_list[layer] != 0:
                self.layers[layer].append(nn.Dropout(dropout_list[layer]))

    def organize_filters(self, filter_org, filter_number, layers):
        """
        Method to organize filter numbers for convolutional layers.

        Args:
        - filter_org (str): Organization pattern for filter numbers.
        - filter_number (int): Number of initial filters.
        - layers (int): Number of convolutional layers.

        Returns:
        - filter_num (list): List of filter numbers for each layer.
        """
        if filter_org == "same":
            filter_num = [filter_number] * layers
        elif filter_org == "double":
            filter_num = [filter_number * (2 ** i) for i in range(layers)]
        elif filter_org == "half":
            filter_num = [int(filter_number * (2 ** (-i))) for i in range(layers)]
        elif filter_org == "alternating_list":
            filter_num = [filter_number if i % 2 == 0 else filter_number * 2 for i in range(layers)]
        elif filter_org == "d_alternating_list":
            filter_num = [filter_number if i % 4 == 0 or i % 4 == 1 else filter_number * 2 for i in range(layers)]
        elif filter_org == 'desc':
            filter_num = [filter_number - i for i in range(layers)]
        elif filter_org == "asc":
            filter_num = [filter_number + i for i in range(layers)]
        return filter_num

    def activation_fun(self, act):
        """
        Method to get activation function module.

        Args:
        - act (str): Name of the activation function.

        Returns:
        - act_fun (nn.Module): Activation function module.
        """
        if act == "ReLU":
            act_fun = nn.ReLU()
        elif act == "GELU":
            act_fun = nn.GELU()
        elif act == "SiLU":
            act_fun = nn.SiLU()
        elif act == "Mish":
            act_fun = nn.Mish()
        elif act == "softmax":
            act_fun = nn.Softmax(dim=1)
        elif act == "ELU":
            act_fun = nn.ELU()
        elif act == "LeakyReLU":
            act_fun = nn.LeakyReLU()
        return act_fun

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x (torch.Tensor): Output tensor.
        """
        for i in range(0, self.con_layers):
            x = self.layers[i](x)  # Pass through convolutional layers
        x = self.flatten(x)  # Flatten the output tensor
        for i in range(0, self.den_layers):
            x = self.layers[i + self.con_layers](x)  # Pass through dense layers
        x = self.layers[self.con_layers + self.den_layers](x)  # Pass through output layer
        return x

    def set_optimizer(self, optimizer_name, learning_rate):
        """
        Method to set the optimizer for the model.

        Args:
        - optimizer_name (str): Name of the optimizer ('SGD', 'Adam', .).
        - learning_rate (float): Learning rate for the optimizer.
        """
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum= arguments.momentum, weight_decay=arguments.weight_decay)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=arguments.weight_decay)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=arguments.weight_decay, momentum=arguments.momentum)
        elif optimizer_name == "nadam":
            self.optimizer = optim.NAdam(self.parameters(), lr = learning_rate, weight_decay=arguments.weight_decay)
        elif optimizer_name == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr = learning_rate, weight_decay=arguments.weight_decay)
        return self.optimizer


#Model Training
def train_model(model, device, PARAM, console_log, wandb_log, return_model=0):
    """
    Function to train the model.

    Args:
    - model (ConvolutionalNeuralNetwork): The model to be trained.
    - device (torch.device): Device to run the training on (CPU or GPU).
    - PARAM (dict): Dictionary containing training parameters.
    - console_log (int): Flag indicating whether to log training details to the console (1 for yes, 0 for no).
    - wandb_log (int): Flag indicating whether to log training details to wandb (1 for yes, 0 for no).
    - return_model (int): Flag indicating whether to return the trained model (1 for yes, 0 for no).

    Returns:
    - (float or ConvolutionalNeuralNetwork): If return_model is 1, returns the trained model.
                                            Otherwise, returns the validation accuracy.
    """

    # Initialize wandb project and set run name if wandb_log is enabled
    if wandb_log == 1:
        wandb.init(project=arguments.wand_project)
        wandb.run.name = 'SAMPLE-RUN'

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = model.set_optimizer(PARAM["optimizer_name"], PARAM["eta"])

    # Load data for training, validation, and testing
    labels, train_loader, val_loader, test_loader = load_data(PARAM["batch_size"], PARAM["image_size"], PARAM["augmentation"])

    # Iterate over epochs
    for epoch in range(PARAM["epochs"]):
        model.train() # Set model to training mode
        # Initialize varibles to 0 for each epochs
        running_loss = 0.0
        correct = 0
        total = 0
        count = 0

        # Training loop
        for images, labels in train_loader: 
            images = images.to(device) # Move image to respective device
            labels = labels.to(device) # Move labels to respective device 
            optimizer.zero_grad()
            outputs = model(images)  # Forward Pass : Compute predicted outputs
            loss = criterion(outputs, labels) # Compute Loss
            loss.backward()  # Backward Pass 
            optimizer.step()  # update parametes 
            running_loss += loss.item()  # Add loss for curr batch to global loss 

            # Calculate training accuracy
            prob, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # For Debuging 
            # count += 1
            # if count % 5 == 0:
            #     print(count, end=" | ")
        # print("")
 
        model.eval() # Set model to evalution mode 
        running_val_loss = 0.0 
        correct_pred = 0
        total_pred = 0

        # Validation loop
        with torch.no_grad():
            for val_img, val_label in val_loader:
                # Move images and labels to device
                val_img = val_img.to(device)
                val_label = val_label.to(device)

                # Forward Pass 
                val_output = model(val_img)

                # Calculating Loss 
                loss_val = criterion(val_output, val_label)
                running_val_loss += loss_val.item()

                #Calculation of Correctly Predicted Class
                prob, class_ = torch.max(val_output.data, 1)
                total_pred += val_label.size(0)
                correct_pred += (class_ == val_label).sum().item()

        # Log training details to console
        if console_log == 1:
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Training_Accuracy: {100 * correct / total}%, Validation_Loss : {running_val_loss/len(val_loader)}, Validation_accuracy : {100 * correct_pred / total_pred}%")

        # Log training details to wandb
        if wandb_log == 1:
            wandb.log(
                {
                    'epochs': epoch+1,
                    'training_loss': running_loss/len(train_loader),
                    'training_accuracy': 100 * correct / total,
                    'Validation_Loss': running_val_loss/len(val_loader),
                    'Validation_accuracy': 100 * correct_pred / total_pred
                }
            )

    # Finish wandb run
    if wandb_log == 1:
        wandb.finish()

    # Return either trained model or validation accuracy based on return_model flag
    if return_model == 1:
        return model
    return 100 * correct_pred / total_pred

## Function to evaluate model
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


############################### PLOTING GRID ###########################

def generate_random_numbers():
    """
    Generate a list of random numbers.

    Returns:
        list: A list of random numbers.
    """
    numbers = []
    for _ in range(10):
        lower_bound = 200 * _
        upper_bound = lower_bound + 199
        group = [random.randint(lower_bound, upper_bound) for _ in range(3)]
        numbers.extend(group)
    return numbers

def get_prediction(model, img, device):
    """
    Get predictions from a trained model for a given image.

    Args:
        model (torch.nn.Module): The trained neural network model.
        img (torch.Tensor): Input image tensor.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the predicted probability and the predicted class index.
    """
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(img)
        prob, class_ = torch.max(pred.data, 1)
    return prob, class_


def predict_test_images(model, test_loader, label, device):
    """
    Predict labels for images in a test dataset using a trained model.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        label (list): List of class labels corresponding to class indices.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The accuracy of the predictions.
    """
    total = 0
    count = 0
    for image, lbl in test_loader:
        print("Index:", total, end="  ")
        image = image.to(device)
        lbl = lbl.to(device)
        print("True Label:", label[lbl.item()], end=" | ")
        prob, pred = get_prediction(model, image, device)
        print("Predicted Label:", label[pred.item()])
        if pred == lbl:
            count += 1
        total += 1
    accuracy = count / total if total != 0 else 0
    print(f"Accuracy: {accuracy}")
    return accuracy

        
def plot_grid(model, device, wandb_log=0, console_log=1):
    """
    Plot a grid of images with their true and predicted labels.

    Args:
        model (torch.nn.Module): The trained neural network model.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        wandb_log (int, optional): Flag to log the plot to Weights & Biases. Defaults to 0.
        console_log (int, optional): Flag to print additional information to the console. Defaults to 1.

    Returns:
        int: Total count of correctly predicted images.
    """

    # Load data
    label, train_loader, val_loader, test_loader = load_data(1, 224, device= device)

    # Generate random indices for selecting images
    random_index = generate_random_numbers()
    print(random_index)
    
    # Initialize variables
    index = 0
    total = 0
    fig, axes = plt.subplots(10, 3, figsize=(10, 30))
    count = 0

    # Iterate through test dataset
    for image, lbl in test_loader:
        # Check if the current index is in the random indices
        if index in random_index:
            image = image.to(device)
            lbl = lbl.to(device)
            lbl.to(int)
            row = count // 3  # Calculating row index for subplot
            col = count % 3   # Calculating column index for subplot
            # Get prediction for the image
            prob, pred = get_prediction(model, image, device)
            # Convert image tensor to numpy array and rearrange dimensions for display
            image = image.cpu().squeeze().permute(1, 2, 0)

            # Display image
            axes[row, col].imshow(image)
            # Set title with true label
            axes[row, col].set_title(label[lbl], fontsize=14, fontweight='bold', color='blue', family='serif', loc='center', pad=8)
            axes[row, col].axis('off')  # Turn off axis
            # Check if prediction is correct and annotate with predicted label
            if pred == lbl:
                total += 1
                axes[row, col].text(0.5, -0.08, label[pred], fontweight='bold', horizontalalignment='center', verticalalignment='center', fontsize=14, color='green', family='serif', transform=axes[row, col].transAxes)
            else:
                axes[row, col].text(0.5, -0.08, label[pred], fontweight='bold', horizontalalignment='center', verticalalignment='center', fontsize=14, color='red', family='serif', transform=axes[row, col].transAxes)
            count += 1
        index += 1
    plt.figtext(0.5, 0, f"Total correctly predicted images: {total}/30", ha='center', fontsize=14, va = 'center', color = 'black', family = 'serif', fontweight = 'bold')
    plt.subplots_adjust(hspace=0.4)  # Adjust spacing between subplots
    plt.tight_layout()  # Adjust layout to prevent overlapping
    
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Log plot to Weights & Biases if enabled
    if wandb_log == 1:
        wandb.init(project=arguments.wandb_project, name='PartA Q4: Grid 10 X 3')
        wandb.log({'Grid 10 X 3': wandb.Image(plt)})
        wandb.finish()

    # Display the plot
    if console_log == 1:
        plt.show()

PARAM = {
    "con_layers" : 5,
    "dense_layers" : 1,
    "output_activation" : "softmax", 
    "input_channel" : 3,
    "padding" : arguments.conv_padding,
    "stride" : arguments.conv_stride,
    "pool_padding" : arguments.pool_padding,
    "pool_stride" : arguments.pool_stride,
    "image_size" : arguments.image_size,
    "pool_filter_size" : arguments.pool_filter_size,
    "filter_size" : arguments.filter_size, 
    "dense_output_list" : [arguments.neurons_fc],
    "filter_num" : arguments.filter_num,
    "activation" : arguments.activation,
    "filter_org" : arguments.filter_org,  #double half
    "batch_size" : arguments.batch_size,
    "eta" : arguments.learning_rate,
    "dropout" : [0, 0, 0, 0, 0] + [arguments.dropout],
    "epochs" : arguments.epochs,
    "augmentation" : arguments.augmentation,
    "batch_norm" : arguments.batch_norm,
    "init" : arguments.weight_init,
    "optimizer_name" : arguments.optimizer
}

# Function to determine and set the device for computation (CPU/GPU)
def set_device():
    device = "cpu"  # Defaulting to CPU
    if torch.cuda.is_available():  # Checking if GPU is available
        device = torch.device("cuda")  # Setting device to GPU if available
    else:
        device = torch.device("cpu")  # Otherwise, default to CPU
    return device


# final call
device = set_device()  # Calling the function to set the device
print("Currently Using :: ", device)  # Printing the currently used device

model = ConvolutionalNeuralNetwork(PARAM)
model = model.to(device)
net = train_model(model, device, PARAM, arguments.console, arguments.wandb_log, 1)
if arguments.plot_grid == 1:
    plot_grid(net, device, wandb_log=arguments.wandb_log, console_log=arguments.console)
if arguments.evaluate == 1:
    calculate_accuracy_on_test_data(net, device)
