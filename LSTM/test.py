# test.py

# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import LSTMClassifier
import torch.nn.functional as F
import os
import pandas as pd
from dataloader import CSVDataset
from sklearn.metrics import accuracy_score, confusion_matrix

# 假设您的TF-IDF矩阵和标签如下
# 这里需要导入或生成您的TF-IDF矩阵和标签


csv_files_test = ['source/train_data/sample/2017.csv', 'source/train_data/sample/2018.csv', 'source/train_data/sample/2019.csv',
                  'source/train_data/sample/2020.csv', 'source/train_data/sample/2021.csv', 'source/train_data/sample/2022.csv']
label_files_test = ['source/train_data/label/label_2017.csv', 'source/train_data/label/label_2018.csv', 'source/train_data/label/label_2019.csv',
                    'source/train_data/label/label_2020.csv', 'source/train_data/label/label_2021.csv', 'source/train_data/label/label_2022.csv']


model = LSTMClassifier(input_dim=19, hidden_dim=64, output_dim=2, num_layers=3)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

test_dataset = CSVDataset(csv_files_test, label_files_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_labels = []
all_predictions = []

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# 计算性能指标
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
