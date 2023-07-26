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

# load data
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# get the stopword
stop_words = set(stopwords.words('english'))

def clean_text(text):
    #Converts all characters in text to lowercase.
    text = text.lower()

    #converts all characters in text to lowercase
    #replace all non-word characters (characters that are not a letter, digit, or underscore) in text with a space.
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)

    words = word_tokenize(text)
    #split the text into individual words.

    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

train_data['text'] = train_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)
print("finish cleaning dataset")

###########################################
#Doc2vec
#https://arxiv.org/abs/1405.4053 Distributed Representations of Sentences and Documents
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
model = Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_documents)
model.train(train_documents, total_examples=model.corpus_count, epochs=model.epochs)

# 使用 Doc2Vec 模型获取向量s
train_vectors = [model.infer_vector(doc.words) for doc in train_documents]
test_vectors = [model.infer_vector(word_tokenize(doc)) for doc in test_data['text']]

# 转换为张量
train_vectors = torch.tensor(train_vectors)
train_vectors = train_vectors.reshape(-1,1,50)
test_vectors = torch.tensor(test_vectors)
test_vectors = test_vectors.reshape(-1,1,50)

print(train_vectors.shape,test_vectors.shape)

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
#https://en.wikipedia.org/wiki/Tf–idf#cite_note-1
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

print(train_vectors.shape,test_vectors.shape)
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
#https://arxiv.org/abs/1301.3781 Efficient Estimation of Word Representations in Vector Space

import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


train_sentences = train_data['text'].apply(word_tokenize).tolist()
test_sentences = test_data['text'].apply(word_tokenize).tolist()

print("finish seperateing dataset")

# train Word2Vec model
model = Word2Vec(sentences=train_sentences, vector_size=50, window=5, min_count=1, workers=4)
model.save("word2vec_model_all.bin")
#model = Word2Vec.load("word2vec_model_all.bin")
print("finish training dataset")


def get_sentence_vectors(sentences):
    vectors = []
    for sentence in sentences:
        #This line creates a list of word vectors for each word in the sentence 
        #that is in the Word2Vec model's vocabulary.
        sentence_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(sentence_vectors) == 0:
            vectors.append([0] * 50)  # If the sentence doesn't have any words that are in
            # the Word2Vec model's vocabulary, the sentence is represented by a vector of 50 zeros.
        else:
            vectors.append(np.mean(sentence_vectors, axis=0))  # Otherwise, the sentence vector is the average
            # of its word vectors. This vector is then added to the list of sentence vectors.
    return vectors

train_vectors = get_sentence_vectors(train_sentences)
test_vectors = get_sentence_vectors(test_sentences)

# turn to tensor
train_vectors = torch.tensor(train_vectors)
#reshpae
train_vectors = train_vectors.reshape(-1,1,50)
test_vectors = torch.tensor(test_vectors)
test_vectors = test_vectors.reshape(-1,1,50)
# get the label
train_labels = torch.tensor(train_data['stars'].values)
test_labels = torch.tensor(test_data['stars'].values)
#save the training dataset as 'train_vectors.pt' and 'train_labels.pt'  
torch.save(train_vectors, 'train_vectors.pt')
torch.save(train_labels, 'train_labels.pt')
#save the testing dataset as 'test_vectors.pt' and 'test_labels.pt'  
torch.save(test_vectors, 'test_vectors.pt')
torch.save(test_labels, 'test_labels.pt')
