import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned dataset
df = pd.read_csv("./valorant_dataset_cleaned.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 2. Create per-match features
df["kills_per_match"] = df["kills"] / df["matches"]
df["deaths_per_match"] = df["deaths"] / df["matches"]
df["assists_per_match"] = df["assists"] / df["matches"]
df["damage_per_match"] = df["damage"] / df["matches"]
df["damage_received_per_match"] = df["damage_received"] / df["matches"]
df["headshots_per_match"] = df["headshots"] / df["matches"]
df["traded_per_match"] = df["traded"] / df["matches"]

# 3. Create ratio features
df["kd_ratio"] = df["kills"] / df["deaths"].replace(0, np.nan)
df["kd_ratio"] = df["kd_ratio"].fillna(0)

df["headshot_rate"] = df["headshots"] / df["kills"].replace(0, np.nan)
df["headshot_rate"] = df["headshot_rate"].fillna(0)

df["damage_per_death"] = df["damage"] / df["deaths"].replace(0, np.nan)
df["damage_per_death"] = df["damage_per_death"].fillna(0)

# 4. Create rank groups
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

print("\nRank group distribution:")
print(df["rank_group"].value_counts())

# 5. Select features
features = [
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

X = df[features]
y = df["rank_group"]

# 6. Basic EDA
plt.figure(figsize=(6, 4))
sns.countplot(x="rank_group", data=df, order=["Low", "Mid", "High"])
plt.title("Player Count by Rank Group")
plt.xlabel("Rank Group")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="rank_group", y="kd_ratio", data=df, order=["Low", "Mid", "High"])
plt.title("K/D Ratio by Rank Group")
plt.xlabel("Rank Group")
plt.ylabel("K/D Ratio")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="rank_group", y="headshot_rate", data=df, order=["Low", "Mid", "High"])
plt.title("Headshot Rate by Rank Group")
plt.xlabel("Rank Group")
plt.ylabel("Headshot Rate")
plt.tight_layout()
plt.show()

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 8. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "LDA": LinearDiscriminantAnalysis(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# 10. Train and evaluate models
for name, model in models.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Mid", "High"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Mid", "High"],
        yticklabels=["Low", "Mid", "High"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Rank Group")
    plt.ylabel("Actual Rank Group")
    plt.tight_layout()
    plt.show()

# 11. Random Forest feature importance
rf_model = models["Random Forest"]

importances = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()