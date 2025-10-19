import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib, os

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

best = joblib.load("models/best_params.pkl")

model = ElasticNet(**best, max_iter=10000, random_state=0)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Modèle entraîné et sauvegardé.")
