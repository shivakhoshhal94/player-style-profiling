def compute_features(df):
    # Rename long column names to cleaner versions
    df = df.rename(columns={
        'Per 90 Minutes xG': 'xG_per90',
        'Per 90 Minutes xAG': 'xAG_per90',
        'Progression PrgC': 'progressive_carries',
        'Progression PrgP': 'progressive_passes',
        'Progression PrgR': 'progressive_runs'
    })

    # Drop rows with missing values in the selected features
    df = df.dropna(subset=[
        'xG_per90', 'xAG_per90',
        'progressive_carries', 'progressive_passes', 'progressive_runs'
    ])

    return df
