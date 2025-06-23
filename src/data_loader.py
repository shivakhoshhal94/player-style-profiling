import pandas as pd

def load_data(path="data/archive_3/2023-2024/combined_players_stats.csv"):
    return pd.read_csv(path)
