import numpy as np
import pandas as pd
from config import (
    PROCESSED_DIR, SERVE_STATS, ROUND_ORDER, UNRANKED_RANK,
    FORM_HALFLIFE, STATS_HALFLIFE, SURFACES, LEVELS,
    PLAYER_FEATURES, CONTEXT_FEATURES,
)

# Elo names aligned with the winner_<feature> / loser_<feature> convention
ELO_RENAME = {
    "elo_winner_pre_match": "winner_elo",
    "elo_loser_pre_match": "loser_elo",
    "elo_winner_surface_pre_match": "winner_elo_surface",
    "elo_loser_surface_pre_match": "loser_elo_surface",
    "n_matches_winner_pre": "winner_n_matches",
    "n_matches_loser_pre": "loser_n_matches",
}

# Features computed in the long format (subset of PLAYER_FEATURES)
HISTORY_FEATURES = [
    "form", "form_surface",
    "serve_won", "return_won", "first_in", "first_won", "second_won", "bp_saved", "ace_rate", "df_rate",
    "tourney_matches", "tourney_minutes", "days_since_last_tourney",
    "h2h_matches", "h2h_wins",
]


# ---------- Long format (step 7) ----------

def _side_view(df: pd.DataFrame, me: str, opp: str, won: int) -> pd.DataFrame:
    """Each match seen from one player's side. me/opp: 'winner' or 'loser'."""
    view = pd.DataFrame({
        "match_idx": df.index,
        "tourney_id": df["tourney_id"],
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "retirement": df["retirement"],
        "minutes": df["minutes"],
        "player_id": df[f"{me}_id"],
        "opp_id": df[f"{opp}_id"],
        "won": won,
    })
    for s in SERVE_STATS:
        view[s] = df[f"{me[0]}_{s}"]
        view[f"opp_{s}"] = df[f"{opp[0]}_{s}"]
    return view


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: two rows per match, one per player, in chronological order."""
    winners = _side_view(df, "winner", "loser", 1)
    losers = _side_view(df, "loser", "winner", 0)
    long = pd.concat([winners, losers], ignore_index=True)
    long = long.sort_values("match_idx", kind="stable").reset_index(drop=True)

    # Stats from retirements are partial, so they are excluded from the historical averages
    stat_cols = SERVE_STATS + [f"opp_{s}" for s in SERVE_STATS]
    long[stat_cols] = long[stat_cols].astype(float)
    long.loc[long["retirement"], stat_cols] = np.nan
    return long


def past_ewm(long: pd.DataFrame, col: str, halflife: float, by=("player_id",)) -> pd.Series:
    """Exponential average of `col` over the group's PREVIOUS matches (default: player)."""
    return long.groupby(list(by))[col].transform(
        lambda s: s.shift(1).ewm(halflife=halflife).mean()
    )


def attach_to_matches(df: pd.DataFrame, long: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Maps the long-format features back onto the matches, with winner_/loser_ prefixes."""
    w = long.loc[long["won"] == 1].set_index("match_idx")[feature_cols].add_prefix("winner_")
    l = long.loc[long["won"] == 0].set_index("match_idx")[feature_cols].add_prefix("loser_")
    return df.join(w).join(l)


# ---------- Features (step 8) ----------

def add_static_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match attributes known from the dataset: ranking, physical, entry."""
    df = df.copy()
    for side in ("winner", "loser"):
        rank = df[f"{side}_rank"]
        df[f"{side}_unranked"] = rank.isna().astype(float)
        df[f"{side}_log_rank"] = np.log(rank.fillna(UNRANKED_RANK))
        df[f"{side}_log_rank_points"] = np.log1p(df[f"{side}_rank_points"].fillna(0))
        df[f"{side}_ht_missing"] = df[f"{side}_ht"].isna().astype(float)
        df[f"{side}_is_left"] = (df[f"{side}_hand"] == "L").astype(float)
        entry = df[f"{side}_entry"]
        df[f"{side}_is_qualifier"] = entry.isin(["Q", "LL"]).astype(float)
        df[f"{side}_is_wildcard"] = (entry == "WC").astype(float)
        df[f"{side}_is_seeded"] = df[f"{side}_seed"].notna().astype(float)
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Match context: ordinal round, one-hot surface and level."""
    df = df.copy()
    unknown_surfaces = set(df["surface"]) - set(SURFACES)
    assert not unknown_surfaces, f"Unexpected surfaces: {unknown_surfaces}"
    unknown_levels = set(df["tourney_level"]) - set(LEVELS)
    assert not unknown_levels, f"Unexpected levels: {unknown_levels}"

    df["round_ord"] = df["round"].map(ROUND_ORDER)
    for s in SURFACES:
        df[f"surface_{s}"] = (df["surface"] == s).astype(float)
    for lv in LEVELS:
        df[f"level_{lv}"] = (df["tourney_level"] == lv).astype(float)
    return df


def _ratio_parts(long: pd.DataFrame) -> dict:
    """Numerator and denominator of each serve/return statistic."""
    L = long
    return {
        "serve_won": (L["1stWon"] + L["2ndWon"], L["svpt"]),
        "return_won": (L["opp_svpt"] - L["opp_1stWon"] - L["opp_2ndWon"], L["opp_svpt"]),
        "first_in": (L["1stIn"], L["svpt"]),
        "first_won": (L["1stWon"], L["1stIn"]),
        "second_won": (L["2ndWon"], L["svpt"] - L["1stIn"]),
        "bp_saved": (L["bpSaved"], L["bpFaced"]),
        "ace_rate": (L["ace"], L["svpt"]),
        "df_rate": (L["df"], L["svpt"]),
    }


def add_history_features(long: pd.DataFrame) -> pd.DataFrame:
    """Historical features computed only on each player's previous matches."""
    long = long.copy()

    # Form: exponential win rate, overall and per surface
    long["form"] = past_ewm(long, "won", FORM_HALFLIFE)
    long["form_surface"] = past_ewm(long, "won", FORM_HALFLIFE, by=("player_id", "surface"))

    # Serve and return: separate EWMs of numerator and denominator, then the ratio
    for name, (num, den) in _ratio_parts(long).items():
        missing = num.isna() | den.isna()
        long["_num"] = num.mask(missing)
        long["_den"] = den.mask(missing)
        ratio = past_ewm(long, "_num", STATS_HALFLIFE) / past_ewm(long, "_den", STATS_HALFLIFE)
        long[name] = ratio.replace([np.inf, -np.inf], np.nan)
    long = long.drop(columns=["_num", "_den"])

    # Fatigue in the current tournament: matches and minutes already played
    long["tourney_matches"] = long.groupby(["player_id", "tourney_id"]).cumcount()
    minutes = long["minutes"].fillna(0)
    long["tourney_minutes"] = minutes.groupby([long["player_id"], long["tourney_id"]]).cumsum() - minutes

    # Head-to-head: meetings and wins against this opponent, before the match
    g_h2h = long.groupby(["player_id", "opp_id"])
    long["h2h_matches"] = g_h2h.cumcount()
    long["h2h_wins"] = g_h2h["won"].cumsum() - long["won"]

    # Inactivity: days since the previous tournament (capped at 365)
    t = long.drop_duplicates(["player_id", "tourney_id"])[["player_id", "tourney_id", "tourney_date"]].copy()
    t["days_since_last_tourney"] = t.groupby("player_id")["tourney_date"].diff().dt.days
    long = long.merge(
        t[["player_id", "tourney_id", "days_since_last_tourney"]],
        on=["player_id", "tourney_id"], how="left",
    )
    long["days_since_last_tourney"] = long["days_since_last_tourney"].fillna(365).clip(0, 365)
    return long


def feature_columns() -> list[str]:
    """All model input columns."""
    return [f"{side}_{f}" for side in ("winner", "loser") for f in PLAYER_FEATURES] + CONTEXT_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=ELO_RENAME)
    df = add_static_player_features(df)
    df = add_context_features(df)

    long = to_long(df)
    assert len(long) == 2 * len(df)
    long = add_history_features(long)

    first = long.groupby("player_id").head(1)
    assert first["form"].isna().all(), "Leakage: a player's first match already has a history"
    assert (first["h2h_matches"] == 0).all(), "Leakage in the head-to-heads"

    df = attach_to_matches(df, long, HISTORY_FEATURES)

    missing_cols = set(feature_columns()) - set(df.columns)
    assert not missing_cols, f"Missing features: {missing_cols}"
    return df


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    df = build_features(df)
    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    cols = feature_columns()
    print(f"Saved {len(cols)} features for {len(df)} matches")
    print("Share of NaN per feature (top 10):")
    print(df[cols].isna().mean().sort_values(ascending=False).head(10))