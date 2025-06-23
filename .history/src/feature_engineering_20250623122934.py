def compute_per90(df):
    stats = ["passes", "tackles", "key_passes", "xG", "xA"]
    for col in stats:
        df[f"{col}_per90"] = df[col] * 90 / df["minutes"]
    return df.dropna()
