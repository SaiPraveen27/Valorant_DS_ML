import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# 3. Choose target variable
target = "damage_per_match"

# 4. Simple linear regression
# Predict damage_per_match using only kills_per_match
simple_feature = ["kills_per_match"]

X_simple = df[simple_feature]
y = df[target]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

y_pred_simple = simple_model.predict(X_test_s)

print("\n" + "=" * 50)
print("Simple Linear Regression")
print("=" * 50)
print("Feature used:", simple_feature[0])
print("Coefficient:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
print("R2 Score:", r2_score(y_test_s, y_pred_simple))
print("MAE:", mean_absolute_error(y_test_s, y_pred_simple))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, y_pred_simple)))

# Plot simple regression
plt.figure(figsize=(7, 5))
plt.scatter(X_test_s, y_test_s, alpha=0.4, label="Actual players")
plt.plot(
    X_test_s,
    y_pred_simple,
    color="red",
    linewidth=2,
    label="Regression line"
)
plt.title("Simple Linear Regression: Kills per Match vs Damage per Match")
plt.xlabel("Kills per Match")
plt.ylabel("Damage per Match")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Multivariate regression using statsmodels
regression_features = [
    "kills_per_match",
    "deaths_per_match",
    "assists_per_match",
    "headshot_rate",
    "kd_ratio",
    "damage_received_per_match"
]

X_multi = df[regression_features]
y_multi = df[target]

# Add constant term for statsmodels
X_multi_const = sm.add_constant(X_multi)

ols_model = sm.OLS(y_multi, X_multi_const).fit()

print("\n" + "=" * 50)
print("Multivariate Linear Regression - Statsmodels OLS")
print("=" * 50)
print(ols_model.summary())

# 6. Random Forest Regressor
X = df[regression_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf_regressor = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

print("\n" + "=" * 50)
print("Random Forest Regressor")
print("=" * 50)
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# 7. Compare actual vs predicted values
results = pd.DataFrame({
    "Actual Damage per Match": y_test,
    "Predicted Damage per Match": y_pred_rf
})

plt.figure(figsize=(7, 5))
plt.scatter(results["Actual Damage per Match"],
            results["Predicted Damage per Match"],
            alpha=0.4)
plt.plot(
    [results.min().min(), results.max().max()],
    [results.min().min(), results.max().max()],
    color="red",
    linewidth=2
)
plt.title("Random Forest Regressor: Actual vs Predicted")
plt.xlabel("Actual Damage per Match")
plt.ylabel("Predicted Damage per Match")
plt.tight_layout()
plt.show()

# 8. Random Forest feature importance
importances = pd.Series(
    rf_regressor.feature_importances_,
    index=regression_features
).sort_values()

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Random Forest Regressor Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()