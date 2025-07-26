import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load and clean data
# ----------------------------

df = pd.read_csv("composite_data.csv", encoding='latin1')

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

# Remove unnamed or empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ----------------------------
# Step 2: Define features and targets
# ----------------------------

X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df[['Wear_Rate_mm³_Nm', 'hardnessHV', 'COF']]

# ----------------------------
# Step 3: Setup LOOCV and model
# ----------------------------

loo = LeaveOneOut()
model = MultiOutputRegressor(SVR(kernel='rbf', C=100, epsilon=0.01))

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

# ----------------------------
# Step 4: Evaluation
# ----------------------------

target_names = y.columns
print("\n📊 Multi-Output SVR (LOOCV) Results:")
for i, target in enumerate(target_names):
    mae = mean_absolute_error(actual_all[:, i], predicted_all[:, i])
    r2 = r2_score(actual_all[:, i], predicted_all[:, i])
    print(f"→ {target}: MAE = {mae:.6f} | R² = {r2:.6f}")

# ----------------------------
# Step 5: Plot results
# ----------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(target_names):
    axes[i].scatter(actual_all[:, i], predicted_all[:, i], color='orange')
    axes[i].plot([min(actual_all[:, i]), max(actual_all[:, i])],
                 [min(actual_all[:, i]), max(actual_all[:, i])],
                 'r--', label='Ideal')
    axes[i].set_title(f'Actual vs Predicted: {target}')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].legend()

plt.tight_layout()
plt.show()
