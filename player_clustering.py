import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load cleaned dataset
df = pd.read_csv("./valorant_dataset_cleaned.csv")

print("Dataset shape:", df.shape)

# 2. Create per-match features
df["kills_per_match"] = df["kills"] / df["matches"]
df["deaths_per_match"] = df["deaths"] / df["matches"]
df["assists_per_match"] = df["assists"] / df["matches"]
df["damage_per_match"] = df["damage"] / df["matches"]
df["damage_received_per_match"] = df["damage_received"] / df["matches"]
df["headshots_per_match"] = df["headshots"] / df["matches"]
df["traded_per_match"] = df["traded"] / df["matches"]

df["kd_ratio"] = df["kills"] / df["deaths"].replace(0, np.nan)
df["kd_ratio"] = df["kd_ratio"].fillna(0)

df["headshot_rate"] = df["headshots"] / df["kills"].replace(0, np.nan)
df["headshot_rate"] = df["headshot_rate"].fillna(0)

df["damage_per_death"] = df["damage"] / df["deaths"].replace(0, np.nan)
df["damage_per_death"] = df["damage_per_death"].fillna(0)

# 3. Select clustering features
cluster_features = [
    "kills_per_match",
    "deaths_per_match",
    "assists_per_match",
    "damage_per_match",
    "damage_received_per_match",
    "headshots_per_match",
    "traded_per_match",
    "kd_ratio",
    "headshot_rate",
    "damage_per_death"
]

X = df[cluster_features]

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find the best number of K-Means clusters using Elbow Method
inertia_values = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_values, inertia_values, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# 6. Find best k using Silhouette Score
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(7, 4))
plt.plot(k_values, silhouette_scores, marker="o")
plt.title("Silhouette Score for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.tight_layout()
plt.show()

best_k = k_values[np.argmax(silhouette_scores)]
print("Best number of clusters based on Silhouette Score:", best_k)

# 7. Apply final K-Means model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means cluster sizes:")
print(df["kmeans_cluster"].value_counts().sort_index())

# 8. Compare K-Means clusters with actual rank groups
rank_groups = {
    "iron": "Low",
    "bronze": "Low",
    "silver": "Low",
    "gold": "Mid",
    "platinum": "Mid",
    "diamond": "High",
    "ascendant": "High",
    "immortal": "High"
}

df["rank_group"] = df["tier"].map(rank_groups)

plt.figure(figsize=(8, 5))
sns.countplot(
    x="kmeans_cluster",
    hue="rank_group",
    data=df,
    hue_order=["Low", "Mid", "High"]
)
plt.title("K-Means Clusters vs Actual Rank Groups")
plt.xlabel("K-Means Cluster")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()

# 9. Average stats for each K-Means cluster
kmeans_summary = df.groupby("kmeans_cluster")[cluster_features].mean()
print("\nK-Means cluster average statistics:")
print(kmeans_summary)

# 10. Hierarchical clustering dendrogram
# Use a sample because dendrograms become unreadable with thousands of points
sample_df = df.sample(n=300, random_state=42)
sample_X = sample_df[cluster_features]
sample_X_scaled = scaler.transform(sample_X)

linked = linkage(sample_X_scaled, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode="lastp", p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Player Groups")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# 11. Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["agglo_cluster"] = agglo.fit_predict(X_scaled)

print("\nAgglomerative cluster sizes:")
print(df["agglo_cluster"].value_counts().sort_index())

# 12. Compare Agglomerative clusters with rank groups
plt.figure(figsize=(8, 5))
sns.countplot(
    x="agglo_cluster",
    hue="rank_group",
    data=df,
    hue_order=["Low", "Mid", "High"]
)
plt.title("Agglomerative Clusters vs Actual Rank Groups")
plt.xlabel("Agglomerative Cluster")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()

# 13. Average stats for each Agglomerative cluster
agglo_summary = df.groupby("agglo_cluster")[cluster_features].mean()
print("\nAgglomerative cluster average statistics:")
print(agglo_summary)