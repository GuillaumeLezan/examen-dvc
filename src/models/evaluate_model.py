import pandas as pd
import joblib, json, os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
model = joblib.load("models/model.pkl")

y_pred = model.predict(X_test)

preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
os.makedirs("data", exist_ok=True)
preds.to_csv("data/predictions.csv", index=False)

scores = {
    "mse": mean_squared_error(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=2)

print("Scores :", scores)
