# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import LSTMClassifier
import torch.nn.functional as F

# 假设您的TF-IDF矩阵和标签如下
# 这里需要导入或生成您的TF-IDF矩阵和标签
tfidf_matrices = torch.randn(100, 6, 100)  # 示例数据
labels = torch.randint(0, 3, (100,))  # 示例标签（3分类问题）

# 将标签转换为one-hot编码
labels = F.one_hot(labels, num_classes=3).float()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrices, labels, test_size=0.2, random_state=42)

# 创建DataLoaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# 模型参数
input_dim = 100
hidden_dim = 64
output_dim = 3
num_layers = 1

# 实例化模型
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')
