import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def load_from_csv(path='data/user_activity.csv'):

    df = pd.read_csv(path)
    # колонка с активностью
    data = df['user_activity'].values.astype('float32').reshape(-1, 1)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler


class TimeSeriesDS(Dataset):

    def __init__(self, data, window=24):
        self.data = data
        self.window = window

    def __len__(self):
        # чтобы не выйти за границы при взятии y
        return len(self.data) - self.window

    def __getitem__(self, idx):
        # x: окно данных
        x = self.data[idx: idx + self.window]

        # y: следующее значение за окном
        y = self.data[idx + self.window]

        # lstm  ожидает (seq_len, input_size) внутри батча
        return torch.FloatTensor(x), torch.FloatTensor(y)