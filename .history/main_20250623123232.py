from src import data_loader, feature_engineering, clustering_model

def main():
    df = data_loader.load_data()
  

    df = feature_engineering.compute_per90(df)
    print(df.columns.tolist())

    features = ['passes_per90', 'tackles_per90', 'key_passes_per90', 'xG_per90', 'xA_per90']
    df_clustered, model = clustering_model.cluster_players(df, features)

    df_clustered.to_csv("clustered_players.csv", index=False)
    print("✅ Finished. Clustered data saved to clustered_players.csv")

    # Optional: show cluster distribution
    print(df_clustered["cluster"].value_counts())

    # Optional: visualize clusters
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(data=df_clustered, x="xG_per90", y="xA_per90", hue="cluster")
    plt.title("🎯 Player Clusters: xG vs xA per 90")
    plt.show()

if __name__ == "__main__":
    main()
