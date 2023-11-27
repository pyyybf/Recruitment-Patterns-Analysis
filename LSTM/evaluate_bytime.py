import torch
from model import LSTMClassifier
from dataloader import CSVDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

# 模型定义等初始化代码...
input_dim = 20
hidden_dim = 256
output_dim = 1
num_layers = 5


model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)

epoch_num = 200


model.load_state_dict(torch.load(
    f'LSTM/utils/source/model/data_2_layer_5_dim_256/lstm_model_{str(epoch_num)}.pth'))
model.eval()

gradients = []

csv_files = [
    'LSTM/utils/source/train_data_2/sample/2017.csv',
    'LSTM/utils/source/train_data_2/sample/2018.csv',
    'LSTM/utils/source/train_data_2/sample/2019.csv',
    'LSTM/utils/source/train_data_2/sample/2020.csv',
    'LSTM/utils/source/train_data_2/sample/2021.csv',
    'LSTM/utils/source/train_data_2/sample/2022.csv'
]

label_file = 'LSTM/utils/source/train_data_2/label/label.csv'

eval_dataset = CSVDataset(csv_files, label_file)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)


# 加载数据等...

original_data, labels = next(iter(eval_loader))
original_data = original_data.numpy()
original_output = model(torch.tensor(original_data)).detach().numpy()
original_accuracy = accuracy_score(labels.numpy(), original_output.round())

time_steps = original_data.shape[1]
num_features = original_data.shape[2]
feature_importances = np.zeros((time_steps, num_features))

for time_step in range(time_steps):
    for feature_index in range(num_features):
        perturbed_data = original_data.copy()
        # 打乱特定时间步的特征
        np.random.shuffle(perturbed_data[:, time_step, feature_index])
        perturbed_output = model(torch.tensor(perturbed_data)).detach().numpy()
        perturbed_accuracy = accuracy_score(
            labels.numpy(), perturbed_output.round())
        feature_importances[time_step,
                            feature_index] = original_accuracy - perturbed_accuracy

# 现在 feature_importances 有一个独立的重要性分数列表，针对每个时间步
for time_step in range(time_steps):
    sorted_indices = np.argsort(feature_importances[time_step])[::-1]
    print(f"Time Step {time_step} Feature Importances:")
    for idx in sorted_indices:
        print(
            f"Feature {idx} importance: {feature_importances[time_step, idx]}")
