import pandas as pd
from config import START_ELO


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    elo_w_pre = []
    elo_l_pre = []
    rating = {}
    n_matches = {}
    n_w_pre = []
    n_l_pre = []

    all_players = set(df["winner_id"]) | set(df["loser_id"])
    for player in all_players:
        rating[player] = START_ELO
        n_matches[player] = 0

    for row in df.itertuples():
        w, l = row.winner_id, row.loser_id
        elo_w, elo_l = rating[w], rating[l]
        elo_w_pre.append(elo_w)
        elo_l_pre.append(elo_l)
        n_w_pre.append(n_matches[w])
        n_l_pre.append(n_matches[l])

        E_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        E_l = 1 - E_w
        K_w = 250 / (n_matches[w] + 5) ** 0.4
        K_l = 250 / (n_matches[l] + 5) ** 0.4

        rating[w] += K_w * (1 - E_w)
        rating[l] += K_l * (0 - E_l)
        n_matches[w] += 1
        n_matches[l] += 1

    df["elo_winner_pre_match"] = elo_w_pre
    df["elo_loser_pre_match"] = elo_l_pre
    df["n_matches_winner_pre"] = n_w_pre
    df["n_matches_loser_pre"] = n_l_pre

    return df

def compute_surface_elo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    elo_w_pre = []
    elo_l_pre = []
    rating = {}
    n_matches = {}

    all_players = set(df["winner_id"]) | set(df["loser_id"])
    surfaces = df["surface"].unique()
    for player in all_players:
        for surface in surfaces:
            rating[(player, surface)] = START_ELO
            n_matches[(player, surface)] = 0

    for row in df.itertuples():
        w = (row.winner_id, row.surface)
        l = (row.loser_id, row.surface)
        elo_w, elo_l = rating[w], rating[l]
        elo_w_pre.append(elo_w)
        elo_l_pre.append(elo_l)

        E_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        E_l = 1 - E_w
        K_w = 250 / (n_matches[w] + 5) ** 0.4
        K_l = 250 / (n_matches[l] + 5) ** 0.4

        rating[w] += K_w * (1 - E_w)
        rating[l] += K_l * (0 - E_l)
        n_matches[w] += 1
        n_matches[l] += 1

    df["elo_winner_surface_pre_match"] = elo_w_pre
    df["elo_loser_surface_pre_match"] = elo_l_pre
    return df

def main():
    data = {
        # Tournament info
        "tourney_id": ["2024-0410", "2024-0410", "2024-0019", "2024-0019"],
        "tourney_name": [
            "Miami Masters",
            "Miami Masters",
            "Australian Open",
            "Australian Open",
        ],
        "surface": ["Hard", "Hard", "Hard", "Hard"],
        "draw_size": [128, 128, 128, 128],
        "tourney_level": ["M", "M", "G", "G"],  # M = Masters 1000, G = Grand Slam
        "tourney_date": [20240318, 20240318, 20240115, 20240115],  # YYYYMMDD
        "match_num": [101, 102, 201, 202],
        # Winner data
        "winner_id": [206173, 207989, 206173, 207989],
        "winner_seed": [2, 1, 4, 2],
        "winner_entry": [None, None, None, None],
        "winner_name": [
            "Jannik Sinner",
            "Carlos Alcaraz",
            "Jannik Sinner",
            "Carlos Alcaraz",
        ],
        "winner_hand": ["R", "R", "R", "R"],
        "winner_ht": [188, 183, 188, 183],
        "winner_ioc": ["ITA", "ESP", "ITA", "ESP"],
        "winner_age": [22.6, 20.9, 22.4, 20.7],
        # Loser data
        "loser_id": [106421, 106421, 106421, 206173],
        "loser_seed": [3, 3, 3, 4],
        "loser_entry": [None, None, None, None],
        "loser_name": [
            "Daniil Medvedev",
            "Daniil Medvedev",
            "Daniil Medvedev",
            "Jannik Sinner",
        ],
        "loser_hand": ["R", "R", "R", "R"],
        "loser_ht": [198, 198, 198, 188],
        "loser_ioc": ["RUS", "RUS", "RUS", "ITA"],
        "loser_age": [28.1, 28.1, 27.9, 22.4],
        # Match data
        "score": [
            "6-1 6-2",
            "6-4 6-3",
            "3-6 3-6 6-4 6-4 6-3",
            "7-6(5) 6-3 4-6 6-4",
        ],
        "best_of": [3, 3, 5, 5],
        "round": ["SF", "F", "F", "SF"],
        "minutes": [69, 85, 224, 190],
        # Winner serve stats
        "w_ace": [7, 5, 14, 9],
        "w_df": [1, 2, 3, 4],
        "w_svpt": [52, 60, 145, 120],
        "w_1stIn": [35, 39, 90, 78],
        "w_1stWon": [28, 30, 68, 58],
        "w_2ndWon": [11, 12, 31, 24],
        "w_SvGms": [8, 10, 22, 19],
        "w_bpSaved": [2, 1, 6, 4],
        "w_bpFaced": [3, 2, 9, 6],
        # Loser serve stats
        "l_ace": [3, 4, 11, 6],
        "l_df": [6, 3, 5, 2],
        "l_svpt": [58, 65, 150, 118],
        "l_1stIn": [30, 40, 85, 75],
        "l_1stWon": [18, 25, 58, 48],
        "l_2ndWon": [9, 10, 29, 20],
        "l_SvGms": [7, 9, 22, 18],
        "l_bpSaved": [1, 3, 8, 3],
        "l_bpFaced": [5, 6, 12, 7],
        # Ranking
        "winner_rank": [3, 2, 4, 2],
        "winner_rank_points": [8310, 8805, 6490, 8805],
        "loser_rank": [4, 4, 3, 4],
        "loser_rank_points": [7765, 7765, 7555, 6490],
    }

    df_matches = pd.DataFrame(data)

    print(df_matches.info())
    print(
        df_matches[
            ["tourney_name", "winner_name", "loser_name", "score", "round"]
        ]
    )
    df_elo = compute_elo(df=df_matches)
    df_elo_surface = compute_surface_elo(df=df_matches)
    print(df_elo)
    print(df_elo_surface)

if __name__ == "__main__":
    main()