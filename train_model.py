# train_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_excel("Bankruptcy-Prevention.xlsx")
df.columns = df.columns.str.strip()  # Remove spaces

# Encode categorical features and save encoders
feature_columns = df.columns[:-1]  # All except 'class'
target_column = 'class'

encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"models/{col}_encoder.pkl")

# Split features and target
X = df[feature_columns]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, "models/bankruptcy_model.pkl")
print("✅ Model and encoders saved successfully in 'models/' folder.")
