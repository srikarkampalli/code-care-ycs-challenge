# Import XGBoost for the regression model, sklearn for metrics
import xgboost as xgb
import os  # for obtaining paths to export the image files
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # Read CSV
import numpy as np  # RMSE calculation
import joblib  # For exporting the XGBoost Model
import matplotlib.pyplot as plt  # For plotting

# Create the independent and dependent variables
X_VARS = [
    "inpatient_beds_used_7_day_avg",
    "a_adult_hospinpbed_occ_7d_avg",
    "hospconf_flucovid_7d_cov",
    "total_beds_7_day_sum",
    "a_adult_hospbeds_7d_sum",
    "total_beds_7_day_avg",
    "a_adult_hospbeds_7d_avg",
    "a_adult_hospbeds_7d_cov",
    "total_beds_7_day_coverage",
]

Y_VAR = "risk_score"

# Read DataFrame
codecare_df = pd.read_csv("../data/codecare_data.csv")

# MO rows, because missouri will be the test data
mo_rows = codecare_df[codecare_df["state"] == "MO"]
non_mo_rows = codecare_df[codecare_df["state"] != "MO"]

# Features and target
X_train = non_mo_rows[X_VARS]
y_train = non_mo_rows[Y_VAR]

X_test = mo_rows[X_VARS]
y_test = mo_rows[Y_VAR]

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    objective="reg:squarederror",  # for regression
    n_estimators=250,  # number of trees
    learning_rate=0.1,  # step size
    max_depth=3,  # depth of each tree
    subsample=0.8,  # fraction of samples for each tree
    colsample_bytree=0.8,  # fraction of features per tree
    random_state=42,
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
print(f"R^2: {r_squared}")

# Feature importance
xgb.plot_importance(model, max_num_features=10)  # show top 10 features
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "images", "feature_importance.png"))
plt.close()

# Predicted vs Actual Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
)  # perfect prediction line
plt.xlabel("Actual risk_score")
plt.ylabel("Predicted risk_score")
plt.title("Predicted vs Actual risk_score")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(os.getcwd(), "images", "predicted_vs_actual.png")
)  # save as PNG
plt.close()  # close figure to avoid overlap

# Residual Distribution Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(os.getcwd(), "images", "residual_distribution.png")
)  # save as PNG
plt.close()

print("Plots saved as 'predicted_vs_actual.png' and 'residuals_distribution.png'")

# Export the model for future use
joblib.dump(model, "../models/risk_score_model.joblib")
print("Model saved as 'risk_score_model.joblib'")
