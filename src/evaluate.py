import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from config import PROCESSED_DIR, TRAIN_START, UNRANKED_RANK

def accuracy_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return accuracy_score(y_true, (y_prob > 0.5).astype(int))

def log_loss_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return log_loss(y_true, y_prob, labels=[0, 1])

def brier_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)

def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_metric(y_true, y_prob),
        "log_loss": log_loss_metric(y_true, y_prob),
        "brier": brier_metric(y_true, y_prob),
    }

def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "accuracy": accuracy_metric(y_true, y_prob),
        "log_loss": log_loss_metric(y_true, y_prob),
        "brier": brier_metric(y_true, y_prob),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    return out

def elo_prob(r_w: pd.Series, r_l: pd.Series) -> np.ndarray:
    return (1 / (1 + 10 ** ((r_l - r_w) / 400))).to_numpy()