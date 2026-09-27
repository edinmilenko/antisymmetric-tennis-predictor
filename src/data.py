import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, ROUND_ORDER, PRE_MATCH_COLS, POST_MATCH_COLS, FIRST_YEAR, LAST_YEAR
from elo import compute_elo, compute_surface_elo


def load_matches(years) -> pd.DataFrame:
    files = [RAW_DIR / f"atp_matches_{y}.csv" for y in years]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["tourney_level"] != "D") & (df["score"] != "W/O")].copy()
    df["retirement"] = df["score"].str.contains("RET|DEF", na=False)

    unclassified = set(df.columns) - set(PRE_MATCH_COLS) - set(POST_MATCH_COLS)
    assert not unclassified, f"Unclassified columns: {unclassified}"

    unknown_rounds = set(df["round"]) - set(ROUND_ORDER)
    assert not unknown_rounds, f"Unmapped rounds: {unknown_rounds}"

    key_cols = ["winner_id", "loser_id", "tourney_date", "tourney_id", "round", "surface"]
    missing = df[key_cols].isna().sum()
    assert missing.sum() == 0, f"NaN in key columns:\n{missing[missing > 0]}"

    df = df.sort_values(
        by=["tourney_date", "tourney_id", "round"],
        key=lambda col: col.map(ROUND_ORDER) if col.name == "round" else col,
    )
    return df.reset_index(drop=True)

def main() -> None:
    df = load_matches(range(FIRST_YEAR, LAST_YEAR + 1))
    df = clean_matches(df)
    df = compute_elo(df)
    df = compute_surface_elo(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
    print(f"Saved {len(df)} matches to {PROCESSED_DIR / 'matches.parquet'}")


if __name__ == "__main__":
    main()