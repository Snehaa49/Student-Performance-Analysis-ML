import pandas as pd
import numpy as np

df = pd.read_csv("StudentPerformanceFactors.csv")

print("--- First 5 Records ---")
print(df.head(5))

print("\n--- Last 5 Records ---")
print(df.tail(5))

df.fillna("Unavailable", inplace=True)

df.drop_duplicates(inplace=True)

print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))

numeric_data = df.select_dtypes(include=["int64", "float64"])

print("\nMinimum Values:\n", numeric_data.min())
print("\nMaximum Values:\n", numeric_data.max())

z_score = (numeric_data - numeric_data.mean()) / numeric_data.std()

print("\nZ-Score:\n", z_score.head())
df.to_csv("cleaned_data.csv", index=False)

#  DECISION TREE CLASSIFICATION

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  Encode categorical data
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

#  Create classification target
def classify(score):
    if score >= 75:
        return 2   # High
    elif score >= 50:
        return 1   # Medium
    else:
        return 0   # Low

df["Performance"] = df["Exam_Score"].apply(classify)

#  Split dataset
X = df.drop(["Exam_Score", "Performance"], axis=1)
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#  Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Decision Tree Results ---")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))