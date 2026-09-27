import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from config import ROOT, MODELS_DIR, HIDDEN, DROPOUT, SEED
from evaluate import evaluate, elo_prob
from split import make_splits
from model import AntisymmetricNet
from train import predict, to_tensors
from baselines import elo_baselines, fit_logistic, fit_gbm, predict_gbm, diff_features

RESULTS_DIR = ROOT / "results"


def load_net(n_player: int, n_context: int) -> AntisymmetricNet:
    model = AntisymmetricNet(n_player, n_context, hidden=HIDDEN, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / "net.pt"))
    model.eval()
    return model


def elo_mix_prob(split: dict) -> np.ndarray:
    m = split["meta"]
    w_mix = (m["winner_elo"] + m["winner_elo_surface"]) / 2
    l_mix = (m["loser_elo"] + m["loser_elo_surface"]) / 2
    p_w = elo_prob(w_mix, l_mix)
    return np.where(m["a_is_winner"].to_numpy(), p_w, 1 - p_w)


def per_sample_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_diff(y, p_1, p_2, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean log-loss difference (model 1 - model 2) and 95% CI. Negative = model 1 is better."""
    d = per_sample_log_loss(y, p_1) - per_sample_log_loss(y, p_2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def breakdown(split: dict, probs: dict, by: str) -> pd.DataFrame:
    y, groups = split["y"], split["meta"][by].to_numpy()
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        for name, p in probs.items():
            rows.append({by: g, "model": name, "n": int(mask.sum()), **evaluate(y[mask], p[mask])})
    return pd.DataFrame(rows)


def plot_calibration(y: np.ndarray, probs: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("predicted probability that A wins")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration on the 2025 test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(path) -> None:
    h = pd.read_csv(MODELS_DIR / "history.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["epoch"] + 1, h["train_loss"], label="train (with dropout)")
    ax.plot(h["epoch"] + 1, h["val_log_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log-loss")
    ax.set_title("Learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    splits, _ = make_splits()
    train_split, test = splits["train"], splits["test"]
    y = test["y"]

    xa, xb, c, _ = to_tensors(test)
    net = load_net(xa.shape[1], c.shape[1])
    p_net = predict(net, xa, xb, c)

    lr = fit_logistic(train_split)
    p_lr = lr.predict_proba(diff_features(test))[:, 1]
    gbm = fit_gbm(train_split)
    p_gbm = predict_gbm(gbm, test)
    p_mix = elo_mix_prob(test)

    results = elo_baselines(test)
    results["logistic"] = evaluate(y, p_lr)
    results["gbm"] = evaluate(y, p_gbm)
    results["net"] = evaluate(y, p_net)
    table = pd.DataFrame(results).T[["accuracy", "log_loss", "brier", "auc"]]
    print(f"Test 2025 (n = {len(y)}):")
    print(table.round(4))

    print("\nLog-loss difference, net minus other model (95% bootstrap CI; negative = net is better):")
    for name, p_ref in [("elo_mix", p_mix), ("logistic", p_lr), ("gbm", p_gbm)]:
        mean, lo, hi = bootstrap_diff(y, p_net, p_ref)
        print(f"  net - {name:8s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    probs = {"elo_mix": p_mix, "logistic": p_lr, "net": p_net}
    by_surface = breakdown(test, probs, "surface")
    by_level = breakdown(test, probs, "tourney_level")
    print("\nBy surface:")
    print(by_surface.round(4).to_string(index=False))
    print("\nBy tournament level:")
    print(by_level.round(4).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "test_metrics.csv")
    by_surface.to_csv(RESULTS_DIR / "test_by_surface.csv", index=False)
    by_level.to_csv(RESULTS_DIR / "test_by_level.csv", index=False)
    plot_calibration(y, probs, RESULTS_DIR / "calibration.png")
    plot_learning_curve(RESULTS_DIR / "learning_curve.png")
    print(f"\nResults saved to {RESULTS_DIR}")