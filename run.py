import mlflow
from src.config import *
from src.data_loader import load_from_csv, TimeSeriesDS, DataLoader
from src.model import HybridForecaster
from src.trainer import train_model


def main():
    data, scaler = load_from_csv('data/user_activity.csv')

    ds = TimeSeriesDS(data, window=WINDOW_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridForecaster(hidden_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    mlflow.set_experiment("user_behavior_v1")
    with mlflow.start_run():
        train_model(model, loader, criterion, optimizer, EPOCHS)
        mlflow.pytorch.log_model(model, "model_final")


if __name__ == "__main__":
    main()