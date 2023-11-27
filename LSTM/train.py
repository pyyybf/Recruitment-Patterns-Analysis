import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTMClassifier
import torch.nn as nn
from dataloader import CSVDataset
import os

print("working dict: ", os.getcwd())

# CSV文件和标签文件的路径，假设csv_files按年份排序
csv_files = [
    'LSTM/utils/source/train_data_2/sample/2017.csv',
    'LSTM/utils/source/train_data_2/sample/2018.csv',
    'LSTM/utils/source/train_data_2/sample/2019.csv',
    'LSTM/utils/source/train_data_2/sample/2020.csv',
    'LSTM/utils/source/train_data_2/sample/2021.csv',
    'LSTM/utils/source/train_data_2/sample/2022.csv'
]
# 最后一年的标签文件路径
label_file = 'LSTM/utils/source/train_data_2/label/label.csv'

# 加载数据集，现在只使用最后一年的标签
train_dataset = CSVDataset(csv_files, label_file)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型参数
input_dim = 20  # 特征数量，根据您的数据进行调整
hidden_dim = 256  # 隐藏层维度
output_dim = 1  # 输出类别数（对于二分类问题）
num_layers = 5  # LSTM层数

# 实例化模型
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 对于二分类问题，如果output_dim=1
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
save_round = 100
nums = 8
# candidate_epochs = [1000]
current_epoch = 0

for num in range(1, nums+1):
    for epoch in range(save_round):
        current_epoch += 1
        model.train()
        total_loss = 0
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            outputs = model(data)

            # 检查输出是否有NaN
            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at batch {i}, epoch {epoch}")
                continue  # 或者 break，如果你想在发现NaN时停止训练

            # 计算损失
            loss = criterion(outputs.squeeze(), labels)

            # 检查损失是否为NaN
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at batch {i}, epoch {epoch}")
                continue  # 或者 break，如果你想在发现NaN时停止训练

            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 计算平均损失
        average_loss = total_loss / len(train_loader)
        print(
            f"Epoch {current_epoch}/{nums*save_round}, Loss: {average_loss}")

# 保存模型
    model_save_path = f'LSTM/utils/source/model/layer_5_dim_256/lstm_model_{num*save_round}.pth'

    if os.path.exists(model_save_path):
        os.remove(model_save_path)  # 删除已存在的模型文件

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 确保保存模型的目录存在
    torch.save(model.state_dict(), model_save_path)
