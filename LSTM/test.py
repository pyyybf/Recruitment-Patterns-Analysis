# test.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMClassifier
import torch.nn.functional as F

# 加载测试数据
# 这里需要替换为您的实际数据加载逻辑
X_test = torch.randn(20, 6, 100)  # 示例数据
y_test = torch.randint(0, 3, (20,))  # 示例标签（3分类问题）

# 将标签转换为one-hot编码
y_test = F.one_hot(y_test, num_classes=3).float()

# 创建DataLoader
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# 模型参数
input_dim = 100
hidden_dim = 64
output_dim = 3
num_layers = 1

# 加载模型
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
