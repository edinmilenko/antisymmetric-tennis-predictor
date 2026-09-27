import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from config import SEED, UNRANKED_RANK, PLAYER_FEATURES
from evaluate import evaluate, elo_prob
from split import make_splits


def swap(split: dict) -> dict:
    """Same split with A and B swapped and y flipped."""
    out = dict(split)
    out["X_A"], out["X_B"] = split["X_B"], split["X_A"]
    out["y"] = 1 - split["y"]
    return out


def diff_features(split: dict) -> np.ndarray:
    return split["X_A"] - split["X_B"]


def full_features(split: dict) -> np.ndarray:
    return np.hstack([split["X_A"], split["X_B"], split["C"]])


def elo_baselines(split: dict) -> dict:
    """Ranking and Elo evaluated on the same split and with the same A/B orientation."""
    m = split["meta"]
    y = split["y"]
    a_win = m["a_is_winner"].to_numpy()
    results = {}

    w_rank = m["winner_rank"].fillna(UNRANKED_RANK)
    l_rank = m["loser_rank"].fillna(UNRANKED_RANK)
    results["ranking"] = {"accuracy": (w_rank < l_rank).mean()}

    p_w = elo_prob(m["winner_elo"], m["loser_elo"])
    results["elo"] = evaluate(y, np.where(a_win, p_w, 1 - p_w))

    w_mix = (m["winner_elo"] + m["winner_elo_surface"]) / 2
    l_mix = (m["loser_elo"] + m["loser_elo_surface"]) / 2
    p_w = elo_prob(w_mix, l_mix)
    results["elo_mix"] = evaluate(y, np.where(a_win, p_w, 1 - p_w))
    return results


def fit_logistic(train: dict) -> LogisticRegression:
    # Without an intercept, P(A beats B) = sigmoid(w · (x_A - x_B)) is antisymmetric by construction
    model = LogisticRegression(fit_intercept=False, max_iter=1000)
    model.fit(diff_features(train), train["y"])
    return model


def fit_gbm(train: dict) -> HistGradientBoostingClassifier:
    # Train on both orientations: every match seen from both sides
    X = np.vstack([full_features(train), full_features(swap(train))])
    y = np.concatenate([train["y"], 1 - train["y"]])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, early_stopping=False, random_state=SEED
    )
    model.fit(X, y)
    return model


def predict_gbm(model: HistGradientBoostingClassifier, split: dict) -> np.ndarray:
    # Average over the two orientations: guarantees P(A beats B) = 1 - P(B beats A)
    p_ab = model.predict_proba(full_features(split))[:, 1]
    p_ba = model.predict_proba(full_features(swap(split)))[:, 1]
    return (p_ab + (1 - p_ba)) / 2


if __name__ == "__main__":
    splits, _ = make_splits()
    train, val = splits["train"], splits["val"]

    results = elo_baselines(val)

    lr = fit_logistic(train)
    results["logistic"] = evaluate(val["y"], lr.predict_proba(diff_features(val))[:, 1])

    gbm = fit_gbm(train)
    results["gbm"] = evaluate(val["y"], predict_gbm(gbm, val))

    table = pd.DataFrame(results).T[["accuracy", "log_loss", "brier", "auc"]]
    print(table.round(4))

    coefs = pd.Series(lr.coef_[0], index=PLAYER_FEATURES)
    print("\nLargest LR coefficients (in absolute value):")
    print(coefs.sort_values(key=abs, ascending=False).head(10).round(3))