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
#"Deep-Sentiment: Sentiment Analysis Using Ensemble of CNN and Bi-LSTM Models"
#Figure1. Also, you can add three or as much as you want models to increase the accuracy. We just take 2 models as an example

class Net_cnn_lstm(nn.Module):
    def __init__(self):
        super(Net_cnn_lstm, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1,1, bias=True)
        self.Bn1 = nn.BatchNorm1d(32)
        #########################################

        self.bi_lstm1 = nn.LSTM(input_size=50, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.bi_lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        self.conv1 = nn.Conv1d(1, 32, 3, 1,1, bias=True)
        self.Bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(1024, 100, bias=True)
        self.fc2 = nn.Linear(100, 5, bias=True)



    def forward(self, x):
        x = torch.flatten(x, 1)
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)
        x = x.view(-1, 1, 128)
        x = F.relu(self.Bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1,1, bias=True)
        self.Bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 32, 3, 1,1, bias=True)
        self.Bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)


        self.fc1 = nn.Linear(32*12, 5, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)


    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.Bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.Bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        #print(x.shape)
        #x = self.fc1(x)
        x = self.fc1(x)
        return x

class Net_LSTM(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()

        #########################################
        #MLP
        #self.Bn3 = nn.BatchNorm1d(32)
        #self.fc4 = nn.Linear(128, 5, bias=True)
        self.bi_lstm1 = nn.LSTM(input_size=50, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.bi_lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(128, 5, bias=True)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)
        x= self.fc1 (x)
        return x


def test(model1,model2, model3,device, test_loader):
    model1.eval()
    model2.eval()
    model3.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target-1
            output = (model1(data)+model2(data)+model3(data))/3
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct



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
model_cnn_lstm = Net_cnn_lstm().to(device)
model_cnn_lstm.load_state_dict(torch.load('Combination_lstm_CNN.pt'), strict=False)

train_vectors = torch.load('../train_vectors.pt')
train_labels = torch.load('../train_labels.pt')
test_vectors = torch.load('../test_vectors.pt')
test_labels = torch.load('../test_labels.pt')

train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=640, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=640, shuffle=False)

model_cnn = Net_CNN().to(device)
model_cnn.load_state_dict(torch.load('../basline_model/Baseline_CNN.pt'), strict=False)

model_LSTM = Net_LSTM().to(device)
model_LSTM.load_state_dict(torch.load('../basline_model/Baseline_lstm.pt'), strict=False)

test(model_cnn,model_LSTM,model_cnn_lstm, device, test_loader)


