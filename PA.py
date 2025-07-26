import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------
# Load and clean the data
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
# LOOCV with Random Forest
# ----------------------------------------

loo = LeaveOneOut()
model = RandomForestRegressor(n_estimators=100, random_state=42)

actual = []
predicted = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    actual.append(y_test.values[0])
    predicted.append(y_pred[0])

# ----------------------------------------
# Evaluate performance
# ----------------------------------------

mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("\n🌳 Random Forest LOOCV Results:")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# ----------------------------------------
# Plot actual vs predicted
# ----------------------------------------

plt.scatter(actual, predicted, color='green', label='Predicted')
plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Wear Rate")
plt.ylabel("Predicted Wear Rate")
plt.title("Random Forest: Actual vs Predicted Wear Rate (LOOCV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

