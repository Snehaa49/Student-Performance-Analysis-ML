from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "StudentPerformanceFactors.csv"))
df.fillna("Unavailable", inplace=True)
df.drop_duplicates(inplace=True)

le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access"
]

X = df[features]

def classify(score):
    if score >= 75:
        return 2
    elif score >= 50:
        return 1
    else:
        return 0

df["Performance"] = df["Exam_Score"].apply(classify)
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced"),
    param_grid,
    cv=3
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get Inputs
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        sleep = float(request.form['sleep'])
        previous = float(request.form['previous'])

        motivation = le_dict["Motivation_Level"].transform(
            [request.form['motivation']]
        )[0]

        internet = le_dict["Internet_Access"].transform(
            [request.form['internet']]
        )[0]

        input_data = [[hours, attendance, sleep, previous, motivation, internet]]

        ml_prediction = model.predict(input_data)[0]

        score = (hours * 5) + (attendance * 0.3) + (previous * 0.5) + (sleep * 2)

        if score > 120:
            result = "High Performance"
        elif score > 70:
            result = "Medium Performance"
        else:
            result = "Low Performance"

        input_values = [hours, attendance, sleep, previous, motivation, internet]
        total = sum(input_values) + 1e-6
        contribution = [round((i / total) * 100, 2) for i in input_values]

        insights = []

        if hours < 3:
            insights.append("📚 Study hours are too low")
        elif hours > 6:
            insights.append("📚 Strong study consistency")

        if attendance < 60:
            insights.append("📊 Low attendance is affecting performance")
        else:
            insights.append("📊 Attendance is good")

        if sleep < 5:
            insights.append("😴 Poor sleep may reduce performance")
        else:
            insights.append("😴 Healthy sleep schedule")

        if previous < 50:
            insights.append("📉 Weak academic foundation")
        else:
            insights.append("📈 Strong previous performance")

        if result == "Low Performance":
            recommendation = "Increase study time and improve attendance."
        elif result == "Medium Performance":
            recommendation = "Focus on weak areas and consistency."
        else:
            recommendation = "Maintain your performance strategy."

        feature_names = [
            "Study Hours",
            "Attendance",
            "Sleep",
            "Previous Score",
            "Motivation",
            "Internet"
        ]

        top_feature = feature_names[np.argmax(contribution)]

        return render_template(
            "index.html",
            prediction=result,
            insights=insights,
            recommendation=recommendation,
            top_feature=top_feature,
            accuracy=round(accuracy * 100, 2),
            best_params=grid.best_params_,
            features=json.dumps(feature_names),
            importance=json.dumps(contribution)
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)