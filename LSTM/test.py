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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import csv

# 假设您的TF-IDF矩阵和标签如下
# 这里需要导入或生成您的TF-IDF矩阵和标签


csv_files_test = ['LSTM/utils/source/test_data_2/sample/2017.csv', 'LSTM/utils/source/test_data_2/sample/2018.csv', 'LSTM/utils/source/test_data_2/sample/2019.csv',
                  'LSTM/utils/source/test_data_2/sample/2020.csv', 'LSTM/utils/source/test_data_2/sample/2021.csv', 'LSTM/utils/source/test_data_2/sample/2022.csv']
label_files_test = 'LSTM/utils/source/test_data_2/label/label.csv'


for epoch_num in [200]:
    print("Testing epoch: ", epoch_num)

    model = LSTMClassifier(input_dim=20, hidden_dim=256,
                           output_dim=1, num_layers=5)
    model.load_state_dict(torch.load(
        f'LSTM/utils/source/model/data_2_layer_5_dim_256/lstm_model_{str(epoch_num)}.pth'))
    model.eval()

    test_dataset = CSVDataset(csv_files_test, label_files_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            # Squeeze to potentially reduce to 1-dimension
            probabilities = torch.sigmoid(outputs).squeeze()
            # Check if probabilities is a scalar (0-d tensor), and if so, unsqueeze it to become 1-d
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)
            predictions = (probabilities >= 0.5).long()
            # Make sure to move tensors to CPU before converting to numpy
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1_score = f1_score(all_labels, all_predictions)

    print("All predictions: ", all_predictions)

    with open('LSTM/utils/source/test_pred/predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['predictions'])
        writer.writerows(zip(all_predictions))

    # print(f"Accuracy: {accuracy}")
    # print(f"Confusion Matrix:\n{conf_matrix}")
    # print(f"F1 Score: {f1_score}")
    # print(f"Recall: {recall}")
