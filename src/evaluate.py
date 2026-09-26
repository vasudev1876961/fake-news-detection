import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

from src.preprocess import clean_text

# Load
df = pd.read_csv("data/test.csv", sep=';', engine='python')
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Load models
lr = joblib.load("models/lr.pkl")
rf = joblib.load("models/rf.pkl")
dt = joblib.load("models/dt.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

Xv = vectorizer.transform(X)

# Predictions
models = {"LR": lr, "RF": rf, "DT": dt}

for name, model in models.items():
    pred = model.predict(Xv)
    print(f"\n{name} Accuracy:", accuracy_score(y, pred))
    print(classification_report(y, pred))