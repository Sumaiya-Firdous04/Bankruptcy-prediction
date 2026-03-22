import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC

st.title("🏦 Bankruptcy Prediction Dashboard")

# Load Dataset
df = pd.read_excel("Bankruptcy-Prevention.xlsx")
df.columns = df.columns.str.strip()

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Bottom 5 Rows:\n")
st.write(df.tail())

# Dataset info
st.subheader("Dataset Information")
st.write("Shape:", df.shape)
st.write("Data Types")
st.write(df.dtypes)

st.subheader("Duplicate Values")
st.write(df.duplicated().sum())

# Missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Class Distribution")

fig, ax = plt.subplots()
sns.countplot(x='class', data=df, ax=ax)
st.pyplot(fig)


# Feature vs class plots
st.subheader("Feature vs Class Distribution")

for col in df.columns[:-1]:
    fig, ax = plt.subplots()
    sns.countplot(x=col, hue='class', data=df, ax=ax)
    ax.set_title(f"{col} vs Class")
    st.pyplot(fig)


# Correlation Heatmap
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# -----------------------------
# Data Encoding
# -----------------------------

le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('class', axis=1)
y = df['class']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

st.subheader("Model Accuracy Comparison")

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc

st.write(results)

# -----------------------------
# Random Forest Model
# -----------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.subheader("Model Accuracy")
st.success(f"Random Forest Accuracy: {accuracy:.4f}")


st.subheader("Classification Report (Scores)")

report = classification_report(y_test, pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# -----------------------------
# Confusion Matrix
# -----------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Feature Importance")

importance = rf.feature_importances_
features = X.columns

fig, ax = plt.subplots()
sns.barplot(x=importance, y=features, ax=ax)
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

# Save model
joblib.dump(rf, "bankruptcy_model.pkl")
st.success("Random Forest model saved successfully!")
