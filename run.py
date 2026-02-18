import mlflow
import torch
from src.config import DEVICE, WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS, LR, EPOCHS, BATCH_SIZE
from src.data_loader import load_from_csv, TimeSeriesDS, DataLoader
from src.model import HybridForecaster
from src.trainer import train_model


def main():
    X_scaled, y_scaled, scaler_y = load_from_csv('data/features.csv')

    ds = TimeSeriesDS(X_scaled, y_scaled, window=WINDOW_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_scaled.shape[1]

    model = HybridForecaster(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    mlflow.set_experiment("user_behavior_v2")

    with mlflow.start_run():
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("window_size", WINDOW_SIZE)

        train_model(model, loader, criterion, optimizer, EPOCHS)

        mlflow.pytorch.log_model(model, "model_final")
        torch.save(model.state_dict(), "model_final.pt")

        import joblib
        joblib.dump(scaler_y, "scaler_y.joblib")


if __name__ == "__main__":
    main()