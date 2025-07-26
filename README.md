# Multi-Output Machine Learning for Tribological Property Prediction

This project applies machine learning—especially multi-output regression using XGBoost and other models—to predict the **wear rate** and **hardness** of **Rice Husk Ash (RHA) reinforced aluminium matrix composites**.

---

## 🔍 Project Description

The research investigates how reinforcement parameters (like % RHA, particle size, load, and sliding speed) affect the tribological performance of aluminium matrix composites. The dataset was prepared using **Taguchi Design of Experiments**, and machine learning models were trained to predict:

- **Wear Rate (mm³/Nm)**
- **Hardness (BHN)**

---

## 🧠 Models Used

- `XGBoost Regressor`
- `Random Forest`
- `Gradient Boosting`
- `Support Vector Regression (SVR)`
- `K-Nearest Neighbors (KNN)`
- `Passive Aggressive Regressor`
- `Decision Tree`
- `MultiOutputRegressor` Wrappers

---

## 📂 Files in This Repository

| File | Description |
|------|-------------|
| `main.py` | Main script for training and testing ML models |
| `xboost.py` / `XBGFINETURN.py` | XGBoost implementation and hyperparameter tuning |
| `Multi.py` | Multi-output regression handler |
| `composite_data.csv` | Main dataset |
| `fine_tune_et.py` | ExtraTreesRegressor tuning |
| `RFmulti.py`, `KNN.py`, etc. | Other regression models |
| `.idea/` | PyCharm project config (optional) |

---

## 🧪 Features & Target Variables

**Input Features:**
- % Reinforcement (RHA)
- Particle Size (µm)
- Applied Load (N)
- Sliding Speed (m/s)

**Target Variables:**
- Wear Rate
- Hardness

---

## 📊 Results

- Achieved **R² > 0.99** for both targets with XGBoost.
- Demonstrated excellent correlation between predicted and experimental results.
- Suitable for use in predictive material design and optimization.

---

## 📌 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Ahmed2155/multioutput-xgboost-rha-composite-.git
   cd multioutput-xgboost-rha-composite-
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the main script:

bash
Copy
Edit
python main.py
