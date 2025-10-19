import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib, os

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

os.makedirs("data/processed", exist_ok=True)
X_train_s.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_s.to_csv("data/processed/X_test_scaled.csv", index=False)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

print("Mise à l'échelle terminée.")
