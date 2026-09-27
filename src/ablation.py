import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT, PLAYER_FEATURES
from evaluate import evaluate
from split import make_splits
from train import train, predict, to_tensors
from baselines import fit_logistic, diff_features

RESULTS_DIR = ROOT / "results"
SEEDS = [0, 1]

# Order used in the cumulative ablation
GROUPS = {
    "elo": ["elo", "elo_surface", "n_matches"],
    "ranking": ["log_rank", "unranked", "log_rank_points"],
    "forma": ["form", "form_surface"],
    "servizio_risposta": ["serve_won", "return_won", "first_in", "first_won",
                          "second_won", "bp_saved", "ace_rate", "df_rate"],
    "fatica_attivita": ["tourney_matches", "tourney_minutes", "days_since_last_tourney"],
    "fisico_entry": ["age", "ht", "ht_missing", "is_left", "is_qualifier", "is_wildcard", "is_seeded"],
    "h2h": ["h2h_matches", "h2h_wins"],
}

# Display labels for the groups in the plot and CSVs
GROUP_LABELS = {
    "elo": "Elo & experience",
    "ranking": "ranking",
    "forma": "form",
    "servizio_risposta": "serve & return",
    "fatica_attivita": "fatigue & activity",
    "fisico_entry": "physical & entry",
    "h2h": "head-to-head",
}


def subset(split: dict, features: list[str]) -> dict:
    """Same split with only some of the player features (the context is kept whole)."""
    idx = [PLAYER_FEATURES.index(f) for f in features]
    out = dict(split)
    out["X_A"] = split["X_A"][:, idx]
    out["X_B"] = split["X_B"][:, idx]
    return out


def run_logistic(splits: dict, features: list[str]) -> dict:
    tr, va = subset(splits["train"], features), subset(splits["val"], features)
    model = fit_logistic(tr)
    return evaluate(va["y"], model.predict_proba(diff_features(va))[:, 1])


def run_net(splits: dict, features: list[str]) -> dict:
    sub = {name: subset(splits[name], features) for name in ("train", "val")}
    xa, xb, c, y = to_tensors(sub["val"])
    runs = []
    for seed in SEEDS:
        model, _ = train(sub, seed=seed, verbose=False)
        runs.append(evaluate(y.numpy(), predict(model, xa, xb, c)))
    runs = pd.DataFrame(runs)
    return {**runs.mean().to_dict(), "log_loss_std": runs["log_loss"].std()}


def plot_ablation(cumulative: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cumulative))
    ax.errorbar(x, cumulative["net_log_loss"], yerr=cumulative["net_log_loss_std"],
                marker="o", capsize=3, label="network (mean over 2 seeds)")
    ax.plot(x, cumulative["lr_log_loss"], marker="s", label="logistic regression")
    ax.set_xticks(x)
    ax.set_xticklabels(cumulative["step"], rotation=30, ha="right")
    ax.set_ylabel("validation log-loss")
    ax.set_title("Cumulative ablation of feature groups")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    splits, _ = make_splits()
    all_features = [f for feats in GROUPS.values() for f in feats]
    assert sorted(all_features) == sorted(PLAYER_FEATURES), "The groups do not cover exactly all the features"

    # 1. Cumulative ablation
    rows, used = [], []
    for i, (group, feats) in enumerate(GROUPS.items()):
        used = used + feats
        label = GROUP_LABELS[group] if i == 0 else f"+ {GROUP_LABELS[group]}"
        lr_m = run_logistic(splits, used)
        net_m = run_net(splits, used)
        rows.append({
            "step": label, "n_feature": len(used),
            "lr_log_loss": lr_m["log_loss"], "lr_accuracy": lr_m["accuracy"],
            "net_log_loss": net_m["log_loss"], "net_log_loss_std": net_m["log_loss_std"],
            "net_accuracy": net_m["accuracy"],
        })
        print(f"{label:22s} LR {lr_m['log_loss']:.4f}   net {net_m['log_loss']:.4f}")
    cumulative = pd.DataFrame(rows)

    # 2. Leave-one-group-out (logistic regression only)
    full = run_logistic(splits, PLAYER_FEATURES)["log_loss"]
    loo = []
    for group, feats in GROUPS.items():
        kept = [f for f in PLAYER_FEATURES if f not in feats]
        loss = run_logistic(splits, kept)["log_loss"]
        loo.append({"group_removed": GROUP_LABELS[group], "lr_log_loss": loss, "delta": loss - full})
    loo = pd.DataFrame(loo).sort_values("delta", ascending=False)

    print("\nCumulative ablation (validation 2024):")
    print(cumulative.round(4).to_string(index=False))
    print(f"\nLeave-one-group-out (LR with all features: {full:.4f}; delta > 0 = the group helps):")
    print(loo.round(4).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    cumulative.to_csv(RESULTS_DIR / "ablation_cumulative.csv", index=False)
    loo.to_csv(RESULTS_DIR / "ablation_leave_one_out.csv", index=False)
    plot_ablation(cumulative, RESULTS_DIR / "ablation.png")
    print(f"\nResults saved to {RESULTS_DIR}")