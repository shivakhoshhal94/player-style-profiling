# ⚽ Football Player Clustering

This project performs **unsupervised clustering of professional football players** using their match statistics (e.g. passes, tackles, xG, xA), specifically focusing on **per 90-minute contributions**. The aim is to **group players into meaningful playstyle roles** without supervision, using real player performance data.

---

## 📁 Project Structure

```
player_clustering_py_version/
├── main.py
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── clustering_model.py
```

---

## 📌 What Each File Does

### `main.py`
This is the main script that runs the entire pipeline:
1. Loads the player dataset
2. Computes per-90 statistics (normalizing for minutes played)
3. Clusters players using KMeans
4. Saves the result to `clustered_players.csv`

### `src/data_loader.py`
Contains a simple function:
```python
load_data(path)
```
It loads the dataset using Pandas. The default path is:
```
data/archive_3/2023-2024/combined_players_stats.csv
```

### `src/feature_engineering.py`
Defines `compute_per90(df)` to generate per-90-minute versions of:
- Passes
- Tackles
- Key passes
- Expected Goals (xG)
- Expected Assists (xA)

### `src/clustering_model.py`
Defines the clustering function:
```python
cluster_players(df, features, n_clusters)
```
- Scales features with `StandardScaler`
- Clusters players using `KMeans`
- Returns the dataframe with a new `cluster` column

---

## 📊 Example Features Used for Clustering
- `passes_per90`
- `tackles_per90`
- `key_passes_per90`
- `xG_per90`
- `xA_per90`

---

## 📁 Output
The result is saved to:
```
clustered_players.csv
```
Each player is tagged with a `cluster` ID. You can explore which players share similar styles statistically.

---

## 🧠 Goal
This project helps demonstrate:
- Feature engineering with sports data
- Unsupervised learning (KMeans)
- Workflow modularization in Python

---

## 📎 Note
Make sure the dataset exists at the correct path:
```
data/archive_3/2023-2024/combined_players_stats.csv
```

---

## 🔜 Ideas to Extend
- Add UMAP or PCA for cluster visualization
- Use DBSCAN or Agglomerative clustering
- Label each cluster (e.g. "Ball Winner", "Playmaker")
- Combine with salary data to find undervalued players

---
