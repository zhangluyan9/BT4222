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
"""
fc1.weight       torch.Size([128, 50])
fc1.bias         torch.Size([128])
Bn1.weight       torch.Size([128])
Bn1.bias         torch.Size([128])
Bn1.running_mean         torch.Size([128])
Bn1.running_var          torch.Size([128])
Bn1.num_batches_tracked          torch.Size([])
fc2.weight       torch.Size([128, 128])
fc2.bias         torch.Size([128])
Bn2.weight       torch.Size([128])
Bn2.bias         torch.Size([128])
Bn2.running_mean         torch.Size([128])
Bn2.running_var          torch.Size([128])
Bn2.num_batches_tracked          torch.Size([])
fc3.weight       torch.Size([5, 128])
fc3.bias         torch.Size([5])
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #########################################
        #MLP

        self.fc1 = nn.Linear(50, 128, bias=True)
        self.Bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.Bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 5, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)



    def forward(self, x):

        x = torch.flatten(x, 1)
        x = F.relu(self.Bn1(self.fc1(x)))
        x = F.relu(self.Bn1(self.fc2(x)))
        x = self.fc3(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        target = target-1
        target = target.long()
        #print(target)
        #print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target-1
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # top1,top5,Bleu, More matrix


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
model = Net().to(device)

if args.resume != None:
    load_model(torch.load(args.resume), model)
for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
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
ACC = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    ACC_ = test(model, device, test_loader)
    if ACC_>ACC or ACC_ == ACC:
        ACC = ACC_
        torch.save(model.state_dict(), "BaseLine_MLP.pt")
    
    scheduler.step()

print(ACC)
#52110/100000

