import torch
import torch.nn as nn


class HybridForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=2):
        super().__init__()

        # ветка LSTM для нелинейных зависимостей
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc_neural = nn.Linear(hidden_dim, 1)

        # ветка трендового прогноза (типа упрощенная ARIMA-like логика)
        self.trend_model = nn.Linear(24, 1)  # 24 - окно

    def forward(self, x):
        # x: (batch, 24, 1)

        # считаем тренд
        trend = self.trend_model(x.squeeze(-1))

        # cчитаем остатки через lstm
        out, _ = self.lstm(x)
        res = self.fc_neural(out[:, -1, :])

        # тренд + корректировка от нейронки
        return trend + res