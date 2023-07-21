import pandas as pd

"""
#STEP1: read the whole dataset
#file_path = 'yelp_academic_dataset_review.json'

#data = pd.read_json(file_path, lines=True)
#selected_columns = data[["text", "stars"]]
#selected_columns.to_csv("yelp_reviews_and_stars.csv", index=False)

"""

"""
#STEP1': incase the dataset is too big
chunksize = 1000000
chunks = pd.read_json(file_path, lines=True, chunksize=chunksize)
i = 0

for chunk in chunks:
    i=i+1
    print(i)
    #print(chunk['text'],chunk['stars'])
    #if i==3:
        #print("3")
        #break
"""


#Step2: generate out dataset for students, each star has 100,000 dataset
data = pd.read_csv("yelp_reviews_and_stars.csv")

selected_data = pd.DataFrame()

for star in range(1, 6):
    selected_reviews = data[data['stars'] == star].sample(n=10000, random_state=1)
    selected_data = pd.concat([selected_data, selected_reviews])

selected_data.to_csv("selected_yelp_reviews_and_stars_small.csv", index=False)



import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split

class YelpDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

"""
Step3: get the training and testing dataset
# 读取数据
data = pd.read_csv("selected_yelp_reviews_and_stars.csv")

# 划分数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 保存训练集和测试集为csv文件
train_data.to_csv("train_dataset.csv", index=False)
test_data.to_csv("test_dataset.csv", index=False)
"""

data = pd.read_csv("selected_yelp_reviews_and_stars_small.csv")

# 初始化训练和测试的DataFrame
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# 对每个星级分别划分训练集和测试集
for star in range(1, 6):
    star_data = data[data['stars'] == star]
    star_train, star_test = train_test_split(star_data, test_size=0.2, random_state=42)
    train_data = pd.concat([train_data, star_train])
    test_data = pd.concat([test_data, star_test])

# 保存训练集和测试集为csv文件
train_data.to_csv("train_dataset_balance_small.csv", index=False)
test_data.to_csv("test_dataset_balance_small.csv", index=False)