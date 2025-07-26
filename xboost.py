import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 1: Load and clean the dataset
# ----------------------------------------

df = pd.read_csv("composite_data_2.csv", encoding='latin1')

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

X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df['Wear_Rate_mm³_Nm']

# ----------------------------------------
# Step 2: Set up XGBoost model and LOOCV
# ----------------------------------------

loo = LeaveOneOut()
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.121,
    max_depth=2,
    random_state=42,
    verbosity=0
)

actual = []
predicted = []

# ----------------------------------------
# Step 3: Run LOOCV
# ----------------------------------------

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    actual.append(y_test.values[0])
    predicted.append(y_pred[0])

# ----------------------------------------
# Step 4: Evaluate performance
# ----------------------------------------

mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("\n🚀 XGBoost Regressor LOOCV Results:")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# ----------------------------------------
# Step 5: Plot Actual vs Predicted
# ----------------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(actual, predicted, color='darkorange', label='Predicted Points')
plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Wear Rate")
plt.ylabel("Predicted Wear Rate")
plt.title("XGBoost: Actual vs Predicted Wear Rate (LOOCV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
