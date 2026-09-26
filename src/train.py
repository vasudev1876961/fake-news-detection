import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import clean_text
from src.feature_engineering import get_vectorizer

# Load dataset (correct path)
df = pd.read_csv("data/train.csv", sep=';', engine='python')
print(df.columns)   # ADD THIS LINE
# Make sure column names match
# If your column is different, change here:
TEXT_COL = "text"
LABEL_COL = "label"

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

X = df[TEXT_COL]
y = df[LABEL_COL]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Vectorization
vectorizer = get_vectorizer()
Xv_train = vectorizer.fit_transform(X_train)

# Models
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=200, max_depth=20)
dt = DecisionTreeClassifier(max_depth=20)

# Train
lr.fit(Xv_train, y_train)
rf.fit(Xv_train, y_train)
dt.fit(Xv_train, y_train)

# Save models (correct path)
joblib.dump(lr, "models/lr.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(dt, "models/dt.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Models trained and saved successfully!")