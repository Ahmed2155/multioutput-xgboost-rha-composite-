import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("composite_data.csv", encoding='latin1')

# Clean and rename columns
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

# Define features and target
X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df['hardnessHV']

# 🎯 Adjust your parameters here
model = XGBRegressor(
    learning_rate=0.121,       # 🔧 Try 0.05 to 0.3
    n_estimators=100,          # 🔧 Try 100 to 500
    max_depth=2,               # 🔧 Try 3 to 6
    min_child_weight=1,        # 🔧 Try 1 to 5
    gamma=1,                   # 🔧 Try 0 to 1
    subsample=0.5,             # 🔧 Try 0.8 to 1.0
    colsample_bytree=1,        # 🔧 Try 0.8 to 1.0
    objective='reg:squarederror',
    random_state=42
)

# Set up Leave-One-Out CV
loo = LeaveOneOut()
actual_values = []
predicted_values = []

# LOOCV loop
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    actual_values.append(y_test.values[0])
    predicted_values.append(y_pred[0])

# Evaluate
actual_values = np.array(actual_values)
predicted_values = np.array(predicted_values)
mae = mean_absolute_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

print("\n📊 XGBoost Hardness Prediction (LOOCV):")
print(f"→ MAE: {mae:.4f}")
print(f"→ R² Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(actual_values, predicted_values, color='green', label='Predicted')
plt.plot([min(actual_values), max(actual_values)],
         [min(actual_values), max(actual_values)], 'r--', label='Ideal')
plt.xlabel('Actual Hardness (HV)')
plt.ylabel('Predicted Hardness (HV)')
plt.title('Actual vs Predicted Hardness (LOOCV - XGBoost)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance
print("\n📈 Feature Importances:")
for feature, score in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {score:.4f}")
