import pandas as pd

def load_data(path=r"F:\projects\player-style-profiling\data\archive_3\2023-2024\combined_players_stats.csv"):
    return pd.read_csv(path)
