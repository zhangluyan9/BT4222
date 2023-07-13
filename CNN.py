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
"""
###########################
#read data
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

train_csv = 'train_dataset_balance_small.csv'
df_train = pd.read_csv(train_csv)
x_train, y_train = df_train['text'].values, df_train['stars'].values

test_csv = 'test_dataset_balance_small.csv'
df_test = pd.read_csv(test_csv)
x_test, y_test = df_test['text'].values, df_test['stars'].values
"""

####################################
#remove stop word
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# 加载数据
train_data = pd.read_csv('train_dataset_balance_small.csv')
test_data = pd.read_csv('test_dataset_balance_small.csv')

# 获取英语停止词
stop_words = set(stopwords.words('english'))

# 定义一个函数来清洗文本
def clean_text(text):
    # 转换为小写
    text = text.lower()
    # 去除符号
    text = re.sub(r'\W', ' ', text)
    # 替换数字
    text = re.sub(r'\d', '', text)
    # 去除停止词
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 清洗数据
train_data['text'] = train_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)

###########################################
#Doc2vec
"""
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

# 加载数据
train_data = pd.read_csv('train_dataset_balance_small.csv')
test_data = pd.read_csv('test_dataset_balance_small.csv')

# 分词
train_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data['text'].apply(word_tokenize))]

# 训练 Doc2Vec 模型
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_documents)
model.train(train_documents, total_examples=model.corpus_count, epochs=model.epochs)

# 使用 Doc2Vec 模型获取向量s
train_vectors = [model.infer_vector(doc.words) for doc in train_documents]
test_vectors = [model.infer_vector(word_tokenize(doc)) for doc in test_data['text']]

# 转换为张量
train_vectors = torch.tensor(train_vectors)
test_vectors = torch.tensor(test_vectors)

# 提取标签
train_labels = torch.tensor(train_data['stars'].values)
test_labels = torch.tensor(test_data['stars'].values)

# 构建 PyTorch 的 Dataset 和 DataLoader
train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
"""

###########################################
#TFIDF

"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 加载数据
#train_data = pd.read_csv('train_dataset_balance_small.csv')
#test_data = pd.read_csv('test_dataset_balance_small.csv')

# 计算TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['text'])
test_vectors = vectorizer.transform(test_data['text'])

# 转换为张量
train_vectors = torch.tensor(train_vectors.toarray())
test_vectors = torch.tensor(test_vectors.toarray())

# 提取标签
train_labels = torch.tensor(train_data['stars'].values)
test_labels = torch.tensor(test_data['stars'].values)

# 构建 PyTorch 的 Dataset 和 DataLoader
train_dataset = torch.utils.data.TensorDataset(train_vectors.float(), train_labels)
test_dataset = torch.utils.data.TensorDataset(test_vectors.float(), test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
"""

#################
#Word2Vec

import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


# 分词
train_sentences = train_data['text'].apply(word_tokenize).tolist()
test_sentences = test_data['text'].apply(word_tokenize).tolist()

# 训练 Word2Vec 模型
model = Word2Vec(sentences=train_sentences, vector_size=50, window=5, min_count=1, workers=4)

# 获取向量
def get_sentence_vectors(sentences):
    vectors = []
    for sentence in sentences:
        sentence_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(sentence_vectors) == 0:
            vectors.append([0] * 50)  # 使用 0 向量作为未知词的向量
        else:
            vectors.append(np.mean(sentence_vectors, axis=0))  # 使用句子中所有词的向量的平均值作为句子的向量
    return vectors

train_vectors = get_sentence_vectors(train_sentences)
test_vectors = get_sentence_vectors(test_sentences)

# 转换为张量
train_vectors = torch.tensor(train_vectors)
train_vectors = train_vectors.reshape(-1,1,50)
test_vectors = torch.tensor(test_vectors)
test_vectors = test_vectors.reshape(-1,1,50)
# 提取标签
train_labels = torch.tensor(train_data['stars'].values)
test_labels = torch.tensor(test_data['stars'].values)
print(test_labels)
# 构建 PyTorch 的 Dataset 和 DataLoader
train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1,1, bias=True)
        self.Bn1 = nn.BatchNorm1d(32)
        #########################################
        #MLP
        self.fc1 = nn.Linear(1600, 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, 128, bias=True)
        self.fc4 = nn.Linear(128, 5, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)

        ###########################################
        #LSTM
        self.fc1 = nn.Linear(1600, 128, bias=True)
        self.bi_lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        self.fc2 = nn.Linear(128, 128, bias=True)
        self.bi_lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        self.fc3 = nn.Linear(128, 64, bias=True)
        self.bi_lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc4 = nn.Linear(64, 5, bias=True)

        ###########################################
        #Attention
        self.self_attn_1 = nn.MultiheadAttention(embed_dim=32, num_heads=8) # Self-attention layer
        self.self_attn_2 = nn.MultiheadAttention(embed_dim=32, num_heads=8) # Self-attention layer
        self.self_attn_3 = nn.MultiheadAttention(embed_dim=32, num_heads=8) # Self-attention layer



        #self.fc2 = nn.Linear(128, 10, bias=True)



    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.Bn1(x)
        x = F.relu(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #print(x.shape)
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
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
#for param_tensor in model.state_dict():
#        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
ACC = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    ACC_ = test(model, device, test_loader)
    if ACC_>ACC or ACC_ == ACC:
        ACC = ACC_
        torch.save(model.state_dict(), "mnist_pretrained.pt")
    
    scheduler.step()

