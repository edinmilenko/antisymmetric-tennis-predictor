from pathlib import Path

FIRST_YEAR = 2005   
TRAIN_START = 2015 
LAST_YEAR = 2025
START_ELO = 1500
UNRANKED_RANK = 10000
SEED = 42
VAL_YEAR = 2024
TEST_YEAR = 2025
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
PROCESSED_DIR = ROOT / "dataset" / "processed"
MODELS_DIR = ROOT / "models"

HIDDEN = 32
DROPOUT = 0.4
LR = 3e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 10

ROUND_ORDER = {
    "RR": 0, "ER": 0.5, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 6.5, "F": 7,
}

SERVE_STATS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
PLAYER_ATTRS = ["seed", "entry", "hand", "ht", "age", "rank", "rank_points"]

ID_COLS = [
    "tourney_id", "tourney_name", "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_ioc",
    "loser_id", "loser_name", "loser_ioc",
]

CONTEXT_COLS = ["surface", "draw_size", "tourney_level", "best_of", "round"]

PLAYER_PRE_MATCH_COLS = [f"{side}_{attr}" for side in ("winner", "loser") for attr in PLAYER_ATTRS]

PRE_MATCH_COLS = ID_COLS + CONTEXT_COLS + PLAYER_PRE_MATCH_COLS

POST_MATCH_COLS = (
    ["score", "minutes", "retirement"]
    + [f"{side}_{stat}" for side in ("w", "l") for stat in SERVE_STATS]
)

FORM_HALFLIFE = 10    # in numero di match
STATS_HALFLIFE = 20
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
LEVELS = ["G", "M", "A", "F", "O"]

# Feature per giocatore: nel DataFrame compaiono con prefisso winner_ / loser_
PLAYER_FEATURES = [
    # Elo ed esperienza
    "elo", "elo_surface", "n_matches",
    # Ranking
    "log_rank", "unranked", "log_rank_points",
    # Attributi fisici ed entry
    "age", "ht", "ht_missing", "is_left", "is_qualifier", "is_wildcard", "is_seeded",
    # Forma
    "form", "form_surface",
    # Servizio e risposta
    "serve_won", "return_won", "first_in", "first_won", "second_won", "bp_saved", "ace_rate", "df_rate",
    # Fatica e attività
    "tourney_matches", "tourney_minutes", "days_since_last_tourney",
    # Head-to-head
    "h2h_matches", "h2h_wins",
]

CONTEXT_FEATURES = (
    ["round_ord", "best_of", "draw_size"]
    + [f"surface_{s}" for s in SURFACES]
    + [f"level_{lv}" for lv in LEVELS]
)