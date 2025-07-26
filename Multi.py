import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ------------------------------------
# STEP 1: Load and clean your dataset
# ------------------------------------

df = pd.read_csv("composite_data.csv", encoding='latin1')  # Replace with your actual file path

# Clean column names
df.columns = (
    df.columns.str.strip()
              .str.replace(' ', '_')
              .str.replace('°', '', regex=False)
              .str.replace('(', '', regex=False)
              .str.replace(')', '', regex=False)
              .str.replace('·', '', regex=False)
              .str.replace('/', '_')
)

# Remove unnecessary unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Optional: Check final columns
print("Final Columns:", df.columns.tolist())

# ------------------------------------
# STEP 2: Define inputs and targets
# ------------------------------------

X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df[['Wear_Rate_mm³_Nm', 'hardnessHV', 'COF']]

# ------------------------------------
# STEP 3: Multi-output regression (LOOCV)
# ------------------------------------

loo = LeaveOneOut()
model = MultiOutputRegressor(XGBRegressor(
    n_estimators=100,
    learning_rate=0.13,
    max_depth=2,
    random_state=42,
    verbosity=0
))

actual_all = []
predicted_all = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    actual_all.append(y_test.values[0])
    predicted_all.append(y_pred[0])

actual_all = np.array(actual_all)
predicted_all = np.array(predicted_all)

# ------------------------------------
# STEP 4: Evaluation for each output
# ------------------------------------

target_names = y.columns
print("\n📊 Multi-Output XGBoost (LOOCV) Results:")
for i, target in enumerate(target_names):
    mae = mean_absolute_error(actual_all[:, i], predicted_all[:, i])
    r2 = r2_score(actual_all[:, i], predicted_all[:, i])
    print(f"→ {target}: MAE = {mae:.6f} | R² = {r2:.6f}")

# ------------------------------------
# STEP 5: Plot Actual vs Predicted for all outputs
# ------------------------------------

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#
# for i, target in enumerate(target_names):
#     axes[i].scatter(actual_all[:, i], predicted_all[:, i], color='teal')
#     axes[i].plot([min(actual_all[:, i]), max(actual_all[:, i])],
#                  [min(actual_all[:, i]), max(actual_all[:, i])],
#                  'r--', label='Ideal')
#     axes[i].set_title(f'Actual vs Predicted: {target}')
#     axes[i].set_xlabel('Actual')
#     axes[i].set_ylabel('Predicted')
#     axes[i].legend()
#
# plt.tight_layout()
# plt.show()
# ----------------------------
# Step 5: Plot results
# ----------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(target_names):
    axes[i].scatter(actual_all[:, i], predicted_all[:, i], color='blue')
    axes[i].plot([min(actual_all[:, i]), max(actual_all[:, i])],
                 [min(actual_all[:, i]), max(actual_all[:, i])],
                 'r--', label='Ideal')
    axes[i].set_title(f'Actual vs Predicted: {target}')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Feature importance
print("\n📈 Feature Importances:")
for feature, score in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {score:.4f}")
