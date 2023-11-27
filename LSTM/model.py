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

        # 只取序列中的最后一个时间步的输出
        # lstm_out的形状是(batch_size, seq_length, hidden_dim)
        last_time_step_out = lstm_out[:, -1, :]

        # 将最后一个时间步的输出传递给全连接层
        out = self.fc(last_time_step_out)

        # out的形状是(batch_size, output_dim)，不需要再次重塑
        return out
