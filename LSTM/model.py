import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM 输出的形状是 (batch_size, seq_length, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # 应用全连接层到每个时间步
        # 这里使用了一个技巧：将三维张量 (batch_size, seq_length, hidden_dim) 重塑为二维张量 (batch_size * seq_length, hidden_dim)，以便能够使用全连接层
        batch_size, seq_length, hidden_dim = lstm_out.shape
        lstm_out = lstm_out.contiguous().view(batch_size * seq_length, hidden_dim)
        out = self.fc(lstm_out)

        # 将输出重塑回 (batch_size, seq_length, output_dim) 的形状
        out = out.view(batch_size, seq_length, -1)

        return out
