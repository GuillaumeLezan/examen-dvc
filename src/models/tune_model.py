import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import joblib, os

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

model = ElasticNet(max_iter=10000, random_state=0)
params = {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}

grid = GridSearchCV(model, params, cv=3)
grid.fit(X_train, y_train)

best = grid.best_params_
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/best_params.pkl")

print("Meilleurs paramètres :", best)
