import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_csv(n_days=90):
    os.makedirs('data', exist_ok=True)

    # генерим временную сетку
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_days * 24)]

    n = len(dates)
    t = np.arange(n)

    # линейный рост (типа юзеров становится больше)
    trend = 0.05 * t

    # суточный цикл (пик вечером, спад ночью)
    seasonality = 20 * np.sin(2 * np.pi * t / 24)

    # шум
    noise = np.random.normal(0, 5, n)

    values = 100 + trend + seasonality + noise

    # добавляем выбросы (типа праздники или наплыва ботов)
    values[np.random.choice(n, int(n * 0.01))] *= 2

    df = pd.DataFrame({'timestamp': dates, 'user_activity': values})

    path = 'data/user_activity.csv'
    df.to_csv(path, index=False)


if __name__ == "__main__":
    generate_csv()