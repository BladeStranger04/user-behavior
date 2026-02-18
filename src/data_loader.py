import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def load_from_csv(path):

    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)

    #data = df['user_activity'].values.astype('float32').reshape(-1, 1)

    y = df['y'].values.astype('float32').reshape(-1, 1)
    X = df.drop(['timestamp', 'y'], axis=1).values.astype('float32')

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)


    return X_scaled, y_scaled, scaler_y

class TimeSeriesDS(Dataset):

    def __init__(self, X, y,window=24):
        self.X = X
        self.y = y
        self.window = window

    def __len__(self):
        # чтобы не выйти за границы при взятии y
        return len(self.X) - self.window

    def __getitem__(self, idx):
        # x: окно данных
        x_seq = self.X[idx : idx + self.window]

        # y: следующее значение за окном
        y_val = self.y[idx + self.window]

        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_val).float()