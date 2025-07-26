import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess
df = pd.read_csv("composite_data.csv", encoding='latin1')
df.columns = (
    df.columns.str.strip()
              .str.replace(' ', '_')
              .str.replace('°', '', regex=False)
              .str.replace('(', '', regex=False)
              .str.replace(')', '', regex=False)
              .str.replace('·', '', regex=False)
              .str.replace('/', '_')
)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("✅ Columns:", df.columns.tolist())

# Select input and target
X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df['hardnessHV']

# 🎯 Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.167,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42
)

# LOOCV
loo = LeaveOneOut()
actual, predicted = [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    actual.append(y_test.values[0])
    predicted.append(pred[0])

# Evaluation
actual = np.array(actual)
predicted = np.array(predicted)

mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("\n📊 Gradient Boosting for Hardness (LOOCV):")
print(f"→ MAE: {mae:.4f}")
print(f"→ R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(actual, predicted, color='purple', label='Predicted')
plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Ideal')
plt.xlabel('Actual Hardness (HV)')
plt.ylabel('Predicted Hardness (HV)')
plt.title('Actual vs Predicted Hardness (LOOCV - GBR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
