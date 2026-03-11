# Bankruptcy Prevention Full Script

# ### Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ### Load Dataset
df = pd.read_excel("Bankruptcy-Prevention.xlsx")

# ✅ Clean column names immediately
df.columns = df.columns.str.strip()
print("Columns:", df.columns)

# ### EDA (Exploratory Data Analysis)
print("Top 5 rows:\n", df.head())
print("Bottom 5 rows:\n", df.tail())
print("Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Duplicate rows:", df.duplicated().sum())
print("Null values:\n", df.isnull().sum())
print("Description:\n", df.describe(include='all'))

# ### Visualization

# 1️⃣ Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.show()

# 2️⃣ Features vs Class
for col in df.columns[:-1]:  # Exclude target 'class'
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='class', data=df)
    plt.title(f"{col} vs Class")
    plt.show()

# 3️⃣ Correlation of numeric features
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix (Numeric Features Only)")
    plt.show()

# ### Transformation: Encode categorical data
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print("After encoding:\n", df.head())

# ### Model Building

# Define features and target
X = df.drop('class', axis=1)
y = df['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1️⃣ Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 2️⃣ Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 3️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4️⃣ KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5️⃣ SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix for last model (SVM)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM)")
plt.show()

# ### Model Comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

print("\nModel Accuracy Comparison:")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: {acc:.4f}")