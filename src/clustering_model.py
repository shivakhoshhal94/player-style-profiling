from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_players(df, features, n_clusters=6):
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans
