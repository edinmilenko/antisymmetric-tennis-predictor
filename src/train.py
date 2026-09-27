import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SEED, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE, MODELS_DIR, HIDDEN, DROPOUT
from evaluate import evaluate, log_loss_metric, accuracy_metric
from model import AntisymmetricNet
from split import make_splits


def to_tensors(split: dict) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(split[k], dtype=torch.float32) for k in ("X_A", "X_B", "C", "y"))


def train_one_epoch(model, loader, loss_fn, optimizer) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x_a, x_b, c, y in loader:
        optimizer.zero_grad()
        logit = model(x_a, x_b, c)
        loss = loss_fn(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def predict(model, x_a, x_b, c) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_a, x_b, c)).numpy()


def train(splits: dict, hidden: int = HIDDEN, dropout: float = DROPOUT, lr: float = LR,
          weight_decay: float = WEIGHT_DECAY, seed: int = SEED,
          verbose: bool = True) -> tuple[AntisymmetricNet, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr, xb_tr, c_tr, y_tr = to_tensors(splits["train"])
    xa_val, xb_val, c_val, y_val = to_tensors(splits["val"])
    y_val_np = y_val.numpy()

    loader = DataLoader(TensorDataset(xa_tr, xb_tr, c_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = AntisymmetricNet(xa_tr.shape[1], c_tr.shape[1], hidden=hidden, dropout=dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_log_loss": []}
    best_loss, best_state, best_epoch, epochs_no_improve = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer)
        p_val = predict(model, xa_val, xb_val, c_val)
        val_loss = log_loss_metric(y_val_np, p_val)
        history["train_loss"].append(train_loss)
        history["val_log_loss"].append(val_loss)
        if verbose:
            val_acc = accuracy_metric(y_val_np, p_val)
            print(f"epoca {epoch:3d}  train loss {train_loss:.4f}  val log-loss {val_loss:.4f}  val acc {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if verbose:
        print(f"Migliore epoca: {best_epoch} (val log-loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    splits, _ = make_splits()
    model, history = train(splits)

    xa_val, xb_val, c_val, y_val = to_tensors(splits["val"])
    p_ab = predict(model, xa_val, xb_val, c_val)
    p_ba = predict(model, xb_val, xa_val, c_val)
    assert np.allclose(p_ab + p_ba, 1, atol=1e-5), "Antisimmetria violata"

    print("\nValidation (pesi migliori):")
    for k, v in evaluate(y_val.numpy(), p_ab).items():
        print(f"  {k:9s} {v:.4f}")
    print("  riferimento regressione logistica: log-loss 0.6061, accuracy 0.6522")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "net.pt")
    pd.DataFrame(history).to_csv(MODELS_DIR / "history.csv", index_label="epoch")
    print(f"Pesi e storico salvati in {MODELS_DIR}")