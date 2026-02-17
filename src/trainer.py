import mlflow
import torch
from tqdm import tqdm
from config import DEVICE


def train_model(model, loader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        mlflow.log_metric("train_loss", total_loss / len(loader), step=epoch)