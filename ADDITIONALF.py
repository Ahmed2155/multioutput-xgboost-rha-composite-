import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset
data = {
    '%_RHA': [5, 5, 5, 10, 10, 10, 15, 15, 15],
    'Sintering_Temp_C': [450, 500, 550, 450, 500, 550, 450, 500, 550],
    'Sintering_Time_min': [60, 90, 120, 90, 120, 60, 120, 60, 90],
    'Sliding_Speed_cm_s': [5, 10, 15, 15, 5, 10, 10, 15, 5],
    'hardnessHV': [44.5, 60.3, 63.3, 82, 89, 90, 93, 109, 111]
}
df = pd.DataFrame(data)

# Features and target
X = df[['%_RHA', 'Sintering_Temp_C', 'Sintering_Time_min', 'Sliding_Speed_cm_s']]
y = df['hardnessHV']

# Model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.13,
    max_depth=2,
    random_state=42,
    verbosity=0
)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_true, y_pred = [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    y_true.extend(y_test)
    y_pred.extend(preds)

# Metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 📊 Feature Importance
# plt.figure(figsize=(8, 5))
plot_importance(model, importance_type='gain', title='Feature Importance (Gain)')
# plt.tight_layout()
plt.show()

# 📉 Residual Analysis
residuals = np.array(y_true) - np.array(y_pred)
plt.figure(figsize=(8, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, color='blue', line_kws={"color": "red", "lw": 1})
plt.xlabel("Predicted Hardness")
plt.ylabel("Residuals")
plt.title("Residual Plot: Hardness Prediction (XGBoost)")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()
