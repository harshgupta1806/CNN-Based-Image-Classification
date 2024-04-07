# CS6910_Assignment2

## PART A

#### CNN Training Script

This script trains a Convolutional Neural Network (CNN) on a given dataset using PyTorch. It allows for configuring various aspects of the CNN architecture, training process, and hyperparameters through command-line arguments.

### Usage

1. Prepare your dataset and update the `datapath` argument before running script.

2. Execute the training script with desired configurations:

### Arguments

The script accepts the following arguments:

| Argument                    | Description                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| -wp / --wandb_project       | Project name used to track experiments in Weights & Biases dashboard. Default is DL-Assignment2.                                                              |
| -we / --wandb_entity       | Wandb Entity used to track experiments in the Weights & Biases dashboard. Default is cs23m026.                                                             |
| -d / --datapath | set the datapath for dataset. type : str |
| -e / --epochs       | No. of epochs used to train the model. Default is 10.                                                   |
| -b / --batch_size         | Batch size used to train neural network. Default is 32.                                        |
| -org / --filter_org              | Defines the organization of filter within Network. Default : same, Choices : same, double, half, asc, desc, alternating_list, d_alternating_list. |
| -f_s / --filter_size             | input list for filter size/kernel size for each layer e.g 11, 9, 7, 5, 3. Default : [11, 9, 7, 5, 3] |
| -f_n / --filter_num   | number of filter in first layer. Default is 64.|
| -pfs / --pool_filter_size        | Size of filter in Pooling layer. Default is 3. |
| -dp / --dropout       | dropout in last layers. Default is 0.1 |
| -aug / --augmentation | Whether to apply data augmentation, Choices : Yes/No, default : Yes |
| -norm / --batch_norm   | Whether to use batch normalization Options: 'Yes', 'No'. Default : 'No'.|
| -img / --image_size   | Image Size e.g. 256 (Image Size = 256 X 256). Default : 256.  |
| -c_p / --conv_padding | Convolution layer padding. Default : 0. |
| -c_s / --conv_stride  | Convolution Layer Stride. Default : 2.   |
| -p_p / --pool_padding  | Pooling Layer padding. Default : 0.|
| -p_s / --pool_stride   | Pooling Layer stride. Default is 1. |
| -o / --optimizer       | Choose Optimizer for training. Choices : "sgd", "rmsprop", "adam", "nadam", "adagrad Default : adam|
| -lr / --learning_rate  | Learning Rate for optimization. Default is 0.0001|
| -m / --momentum       | momentum (for sgd and rmsprop). Default is 0.9 |
| -w_d / --weight_decay  | set weight_decay for regularization, default : 0|
| -w_i / --weight_init   | Initialize weights and baises, default : random, Choices : random, Xavior |
| -ndl / --neurons_fc           | No of neuron in dense layer |
| -a / --activation | set activation function, default : ReLU, Choices : ReLU, LeakyReLU, Mish, GELU, SiLU|
| -p / --console           | Whether to log training metrics on console. Options: 1 (Yes), 0 (No). Default is 1.                                |
| -wl / --wandb_log       | Whether to log training metrics on console. Options: 1 (Yes), 0 (No). Default is 1.|

### ConvolutionalNeuralNetwork Class

This class defines architecture of model, consists of various helper function to make the architecture.


### CNN Architecture

The structure of the CNN is detailed in the `train_partA.ipynb` script, featuring several convolutional layers, each paired with max-pooling layers, and concluding with a fully connected (dense) layer. This setup offers adaptability through the manipulation of different parameters.

### Training Process (Train_Model() method) 

The training process involves optimizing the CNN parameters to minimize the cross-entropy loss between predicted and actual labels. It consists of iterating over the dataset for multiple epochs, updating model parameters using backpropagation, and evaluating the model's performance on validation and test sets.


### Prediction Function

The prediction function `get_prediction` takes an model on which we want to predict, image as input, device (cpu/cuda) as parameters and returns the predicted class label using the trained CNN model (model passed as parameter).

### Plotting Grid

The `plot_grid` function plots a 10 x 3 grid of images with their true labels and predicted labels. It can be used to visualize the model's predictions on a subset of the test dataset.
This function randomly plots 30 images from test data, with their correct as well as predicted label

### Example Usage

```bash
python train_partA.py -d ../path/to/dataset -e 3 --b 64 -org same -f_s 11 9 7 5 3 -f_n 64 -dp 0.3 -aug No -norm Yes -img 224 -o adam -a Mish -p 1 -wb 1 10 
```

## PART B

### Argument Table

| Argument                | Description                                                  | Default Value                                        |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| -d / --datapath          | Give path for dataset | D:/Deep_Learning_A2/nature_12K/inaturalist_12K |
| -e / --epochs          | Number of epochs to train neural network.| 10 |
| -k / --layers_to_freeze | Number of layers to freeze | 17  |                                  
| -s / --strategy         | Layer freezing strategy (choices: "start", "middle", "end", "freeze_all") | "freeze_all"                                              |
| -o / --optimizer       | Optimizer choice (choices: "sgd", "momentum", "nag", "rmsprop", "adam", "nadam") | "adam"                                               |
| -m / --momentum      | Momentum (beta) for the optimizer (only applicable for SGD, rmsprop)  | 0.9                                                  |
| -lr / --learning_rate   | Learning rate for the optimizer  | 0.001   |
| -bs / --batch_size| Batch size for training | 32   |

### Methods Description

#### Identify device, CPU/GPU (`set_device()`)

perameter - None
Discription - This method set the device varible based on availability of CPU or GPU
Return - device type available


#### Data Loading Function (`load_data`):

Loads the dataset from the specified directory and performs necessary transformations such as resizing, normalization, and data augmentation and returns the data loader.
Parameter : batch_size, image_size, device

#### Layer Freezing Function (`freeze_layers`):

Freezes layers based on the specified strategy. It allows freezing the first k layers, middle layers, or the last k layers of the model and freeze all layers except output layer.

#### Train Model Function (`train_model`):

Main function responsible for fine-tuning the GoogleNet model. It loads the data, initializes the model, freezes layers based on the specified strategy, defines the loss function and optimizer, and trains the model using the `Train` function.

### load model Function (`load_model`)

This function return the googlenet model after modification in output layer as per our requirement.

### Example Execution

To execute the script, you can use the following command in the terminal:

```bash
python train_partB.py -e 10 -s "freeze_all" -o "adam" -lr 0.0001 -dp "/path/to/data" -b 32
```

This command will fine-tune the GoogleNet model with the specified arguments for 10 epochs, freezing all layers except output layer, using the adam optimizer with a weight decay of 0, and learning rate of 0.0001. Adjust the arguments as needed for your specific use case.

### References

1. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
2. [Weights & Biases Documentation](https://docs.wandb.ai/)

