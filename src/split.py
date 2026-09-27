import numpy as np
import pandas as pd
from config import (
    PROCESSED_DIR, PLAYER_FEATURES, CONTEXT_FEATURES,
    TRAIN_START, VAL_YEAR, TEST_YEAR, SEED,
)

# Colonne non usate dal modello, conservate per analisi e baseline sul test
META_COLS = [
    "tourney_date", "tourney_name", "surface", "tourney_level", "round",
    "winner_name", "loser_name",
    "winner_elo", "loser_elo", "winner_elo_surface", "loser_elo_surface",
    "winner_rank", "loser_rank",
]


def orient(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Assegna a caso chi è A e chi è B. y = 1 se vince A."""
    X_w = df[[f"winner_{f}" for f in PLAYER_FEATURES]].to_numpy(dtype=float)
    X_l = df[[f"loser_{f}" for f in PLAYER_FEATURES]].to_numpy(dtype=float)
    flip = rng.random(len(df)) < 0.5  # True: A è il perdente
    return {
        "X_A": np.where(flip[:, None], X_l, X_w),
        "X_B": np.where(flip[:, None], X_w, X_l),
        "C": df[CONTEXT_FEATURES].to_numpy(dtype=float),
        "y": (~flip).astype(float),
        "meta": df[META_COLS].assign(a_is_winner=~flip).reset_index(drop=True),
    }


class Preprocessor:
    """Imputazione con mediane e standardizzazione, stimate SOLO sul train.
    A e B condividono gli stessi parametri, perché la rete userà gli stessi pesi per entrambi."""

    def fit(self, split: dict) -> "Preprocessor":
        players = np.vstack([split["X_A"], split["X_B"]])
        self.p_median = np.nanmedian(players, axis=0)
        players = np.where(np.isnan(players), self.p_median, players)
        self.p_mean = players.mean(axis=0)
        self.p_std = players.std(axis=0) + 1e-8

        C = split["C"]
        self.c_median = np.nanmedian(C, axis=0)
        C = np.where(np.isnan(C), self.c_median, C)
        self.c_mean = C.mean(axis=0)
        self.c_std = C.std(axis=0) + 1e-8
        return self

    def transform(self, split: dict) -> dict:
        out = dict(split)
        for key in ("X_A", "X_B"):
            X = np.where(np.isnan(split[key]), self.p_median, split[key])
            out[key] = (X - self.p_mean) / self.p_std
        C = np.where(np.isnan(split["C"]), self.c_median, split["C"])
        out["C"] = (C - self.c_mean) / self.c_std
        return out


def make_splits() -> tuple[dict, Preprocessor]:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    year = df["tourney_date"].dt.year
    samples = df[(year >= TRAIN_START) & (~df["retirement"])]
    year = samples["tourney_date"].dt.year

    parts = {
        "train": samples[year < VAL_YEAR],
        "val": samples[year == VAL_YEAR],
        "test": samples[year == TEST_YEAR],
    }
    rng = np.random.default_rng(SEED)
    splits = {name: orient(part, rng) for name, part in parts.items()}

    prep = Preprocessor().fit(splits["train"])
    splits = {name: prep.transform(s) for name, s in splits.items()}
    return splits, prep


if __name__ == "__main__":
    splits, _ = make_splits()
    for name, s in splits.items():
        n_nan = sum(int(np.isnan(s[k]).sum()) for k in ("X_A", "X_B", "C"))
        years = s["meta"]["tourney_date"].dt.year
        print(f"{name:5s}  n={len(s['y']):6d}  anni {years.min()}-{years.max()}  "
              f"quota A vince={s['y'].mean():.3f}  NaN={n_nan}")