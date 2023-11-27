import torch
from model import LSTMClassifier
from dataloader import CSVDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score
import numpy as np


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

original_data, labels = next(iter(eval_loader))
original_data = original_data.numpy()
original_output = model(torch.tensor(original_data)).detach().numpy()
original_accuracy = accuracy_score(labels.numpy(), original_output.round())


feature_importances = np.zeros(original_data.shape[2])
print(original_data.shape)

for feature_index in range(original_data.shape[2]):
    perturbed_data = original_data.copy()
    np.random.shuffle(perturbed_data[:, :, feature_index])
    perturbed_output = model(torch.tensor(perturbed_data)).detach().numpy()
    perturbed_accuracy = accuracy_score(
        labels.numpy(), perturbed_output.round())
    feature_importances[feature_index] = original_accuracy - perturbed_accuracy

sorted_indices = np.argsort(feature_importances)[::-1]
for idx in sorted_indices:
    print(f"Feature {idx} importance: {feature_importances[idx]}")


# def hook_function(module, grad_in, grad_out):
#     gradients.append(grad_out[0])


# hook = model.lstm.register_full_backward_hook(hook_function)

# batch_data, batch_labels = next(iter(eval_loader))

# outputs = model(batch_data)

# criterion = nn.BCEWithLogitsLoss()
# loss = criterion(outputs.squeeze(), batch_labels)

# model.zero_grad()

# loss.backward()

# # print(gradients)

# batch_gradients = gradients[0]
# print(batch_gradients.shape)


# hook.remove()
