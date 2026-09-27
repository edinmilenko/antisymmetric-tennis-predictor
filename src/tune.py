import itertools
import numpy as np
import pandas as pd

from config import ROOT
from evaluate import evaluate
from split import make_splits
from train import train, predict, to_tensors

GRID = {
    "hidden": [32, 64],
    "dropout": [0.2, 0.4],
    "weight_decay": [1e-4, 1e-3],
    "lr": [1e-3, 3e-4],
}
SEEDS = [0, 1]


if __name__ == "__main__":
    splits, _ = make_splits()
    xa_val, xb_val, c_val, y_val = to_tensors(splits["val"])
    y_val = y_val.numpy()

    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, values in enumerate(combos, 1):
        params = dict(zip(GRID.keys(), values))
        for seed in SEEDS:
            model, history = train(splits, **params, seed=seed, verbose=False)
            metrics = evaluate(y_val, predict(model, xa_val, xb_val, c_val))
            rows.append({**params, "seed": seed, "epochs": len(history["val_log_loss"]), **metrics})
        mean_loss = np.mean([r["log_loss"] for r in rows[-len(SEEDS):]])
        print(f"[{i:2d}/{len(combos)}] {params}  mean log-loss {mean_loss:.4f}")

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(list(GRID.keys()))[["log_loss", "accuracy", "auc"]]
        .agg(["mean", "std"])
        .sort_values(("log_loss", "mean"))
    )
    print("\nTop 10 configurations (mean over seeds):")
    print(summary.round(4).head(10))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    runs.to_csv(results_dir / "tuning_runs.csv", index=False)