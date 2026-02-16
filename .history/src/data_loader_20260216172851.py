import pandas as pd

def load_data(path=r"F:\projects\player-style-profiling\combined_players_stats.csv"):
    return pd.read_csv(path)
