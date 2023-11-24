import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, csv_files, label_file):
        # 初始化数据集
        self.data = []

        # 加载所有特征数据文件
        for file in csv_files:
            data_df = pd.read_csv(file).drop(
                ["cik", "year", "file_name"], axis=1)
            self.data.append(data_df.values)

        # 加载标签文件，假设最后一年的标签在单独的一个文件中
        self.labels = pd.read_csv(label_file).values.squeeze()

    def __len__(self):
        # 假设所有年份的企业数量相同
        return len(self.data[0])

    def __getitem__(self, idx):
        # 提取对应索引的特征，创建时间序列样本
        sample = np.stack([self.data[year][idx]
                          for year in range(len(self.data))], axis=0)

        # 获取对应的标签
        label = self.labels[idx]

        # 将数据和标签转换为张量
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return sample_tensor, label_tensor
