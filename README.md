# Student Performance Analysis and Prediction

A Data Engineering Techniques mini project that explores student performance data, performs data cleaning and statistical analysis, applies a Decision Tree classifier, and provides a Flask-based web interface for performance prediction.

## Project Overview

The project uses the `StudentPerformanceFactors.csv` dataset to examine how factors such as study hours, attendance, sleep, previous scores, motivation, and internet access relate to student exam performance.

It includes two main workflows:

1. **Data analysis and preprocessing** through `project.py`, including missing-value handling, duplicate removal, descriptive statistics, z-score calculation, and a cleaned CSV export.
2. **Interactive prediction application** through `app.py`, which trains a Decision Tree model with `GridSearchCV` and serves a prediction form using Flask.

## Features

- Dataset inspection and preprocessing
- Missing-value handling and duplicate removal
- Mean, median, standard deviation, minimum, and maximum calculations
- Z-score calculation for numeric columns
- Decision Tree classification into Low, Medium, and High performance bands
- Hyperparameter tuning with `GridSearchCV`
- Model accuracy display
- Flask web interface for entering student-related factors and viewing a prediction
- Power BI report and project presentation/report files included

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Web application | Flask, HTML, CSS, JavaScript |
| Visualisation | Chart.js, Power BI |
| Dataset | StudentPerformanceFactors.csv |

## Repository Structure

```text
DET-Mini-Project/
├── app.py
├── project.py
├── StudentPerformanceFactors.csv
├── cleaned_data.csv
├── requirements.txt
├── templates/
│   └── index.html
├── Data Analysis.pptx
├── DET Report.docx
├── final_report_with_lit_review.docx
├── mini_project_report.docx
└── graph.pbix
```

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Replace `USERNAME/REPOSITORY` with your GitHub username and repository name.

### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the data analysis script

```bash
python project.py
```

This script processes the dataset and writes `cleaned_data.csv` in the project directory.

### 5. Run the Flask prediction application

```bash
python app.py
```

Open the local address shown in the terminal (normally `http://127.0.0.1:5000/`) in your browser.

## Model Workflow

1. Load the student performance dataset.
2. Replace missing values and remove duplicate records.
3. Encode categorical variables.
4. Create performance classes from exam scores:
   - High: score >= 75
   - Medium: score >= 50 and < 75
   - Low: score < 50
5. Split the data into training and testing sets.
6. Tune Decision Tree hyperparameters using `GridSearchCV`.
7. Evaluate the selected model using accuracy.
8. Use the trained model to predict a performance class from web-form inputs.

## Notes and Limitations

- This is an educational mini project, not a validated system for making high-stakes academic decisions.
- The performance categories are defined using fixed score thresholds in the code.
- The reported model accuracy depends on the dataset, preprocessing choices, and train/test split.
- The current web application uses a simple rule-based score label alongside the trained classifier; these should not be interpreted as the same prediction method.
- The included Power BI file (`graph.pbix`) can be opened with Power BI Desktop.
- The project uses a public/local CSV dataset. Retain the dataset's original attribution and licence information when redistributing it.

## Project Deliverables

- Python data analysis and preprocessing code
- Flask-based prediction interface
- Original and cleaned datasets
- Power BI dashboard file
- Project presentation and written reports

## Future Improvements

- Use a preprocessing pipeline to avoid fitting categorical encoders on the full dataset before splitting.
- Evaluate the model with precision, recall, F1-score, and a confusion matrix.
- Add input validation and clearer error messages to the web application.
- Improve the dashboard with explainable model outputs and feature importance calculated from the trained model.
- Add tests and deploy the application.

## Academic Project

**Domain:** Data Engineering, Data Analysis, and Machine Learning  
**Project:** Student Performance Analysis and Prediction
