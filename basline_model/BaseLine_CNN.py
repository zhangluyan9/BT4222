import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer


##########################################
#models
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# The network structure was from 
#"Performance Analysis of Different Neural Networks for Sentiment Analysis on IMDb Movie Reviews"
#Figure1. and we set it as the CNN baseline. 

"""
Network Structure
conv1.weight     torch.Size([32, 1, 3])
conv1.bias       torch.Size([32])
Bn1.weight       torch.Size([32])
Bn1.bias         torch.Size([32])
Bn1.running_mean         torch.Size([32])
Bn1.running_var          torch.Size([32])
Bn1.num_batches_tracked          torch.Size([])
conv2.weight     torch.Size([32, 32, 3])
conv2.bias       torch.Size([32])
Bn2.weight       torch.Size([32])
Bn2.bias         torch.Size([32])
Bn2.running_mean         torch.Size([32])
Bn2.running_var          torch.Size([32])
Bn2.num_batches_tracked          torch.Size([])
fc1.weight       torch.Size([5, 416])
fc1.bias         torch.Size([5])
"""

class Net(nn.Module):  # Defines a new neural network architecture as a class that inherits from the PyTorch base class nn.Module.
    def __init__(self):  
        super(Net, self).__init__()  
        self.conv1 = nn.Conv1d(1, 32, 3, 1,1, bias=True)  
        # Define the first 1D convolution layer. Takes 1 input channel, outputs 32 channels, kernel size is 3, stride is 1, padding is 1.
        self.Bn1 = nn.BatchNorm1d(32)  
        # Apply Batch Normalization to the output of the first convolutional layer.
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)  
        # Apply 1D Average Pooling after the first Batch Normalization. The kernel size and stride are 2.

        self.conv2 = nn.Conv1d(32, 32, 3, 1,1, bias=True)  
        self.Bn2 = nn.BatchNorm1d(32)  
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)  

        self.fc1 = nn.Linear(32*12, 5, bias=True)  
        # Define a linear layer (fully connected layer). It takes 32*12 inputs and outputs 5 nodes.


    def forward(self, x):  
        x = F.relu(self.Bn1(self.conv1(x)))  
        # Pass the input through the first convolutional layer, then Batch Normalization, and then apply ReLU activation.
        x = self.pool1(x)  
        # Apply Average Pooling to the output of the previous step.
        x = F.relu(self.Bn2(self.conv2(x)))  
        x = self.pool2(x)  
        x = torch.flatten(x, 1)  
        # Flatten the output from the previous step. This is necessary because fully connected layers expect a 1D input.
        x = self.fc1(x)  
        # Pass the flattened output through the fully connected layer. This is the output of the network.
        return x  


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Set the model to training mode

    for batch_idx, (data, target) in enumerate(train_loader):  # Loop over each batch from the training set
        data, target = data.to(device), target.to(device)  # Move the data to the device that is used

        target = target-1  # Adjust the target values (Moving 1-5 to 0-4  for easy training)
        target = target.long()  # Make sure that target data is long type (necessary for loss function)

        optimizer.zero_grad()  # Clear gradients from the previous training step
        output = model(data)  # Run forward pass (model predictions)

        loss = F.cross_entropy(output, target)  # Calculate the loss between the output and target
        loss.backward()  # Perform backpropagation (calculate gradients of loss w.r.t. parameters)
        optimizer.step()  # Update the model parameters

        if batch_idx % args.log_interval == 0:  # Print log info for specified interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:  # If dry run, stop training after one batch
                break


def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Deactivates autograd, reduces memory usage and speeds up computations
        for data, target in test_loader:  # Loop over each batch from the testing set
            data, target = data.to(device), target.to(device)  # Move the data to the device that is used
            target = target-1  # Adjust the target values
            output = model(data)  # Run forward pass (model predictions)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability as the predicted output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate the average loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct  # Return the number of correctly classified samples


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                    help='Resume model from checkpoint')
                    
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)

if args.resume != None:
    load_model(torch.load(args.resume), model)
for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#Form training and testing dataset
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
train_vectors = torch.load('../train_vectors.pt')
train_labels = torch.load('../train_labels.pt')
test_vectors = torch.load('../test_vectors.pt')
test_labels = torch.load('../test_labels.pt')

train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=640, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=640, shuffle=False)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

#Model training
ACC = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    ACC_ = test(model, device, test_loader)
    if ACC_>ACC or ACC_ == ACC:
        ACC = ACC_
        torch.save(model.state_dict(), "Baseline_CNN.pt")
    
    scheduler.step()

print(ACC)


