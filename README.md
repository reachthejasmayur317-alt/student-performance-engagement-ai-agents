# student-performance-engagement-ai-agents
An end-to-end AI-powered Learning Analytics System designed to predict student performance, classify engagement levels, generate personalized study recommendations, and detect behavioral anomalies.
This project integrates Machine Learning, AI agents, and Power BI dashboards to deliver actionable academic insights.

Project Overview

This project simulates a modern education analytics system using three AI Agents:

Agent 1 — Predictive Analytics
✔ Predictive Model (Regression)

Predicts student quiz performance based on engagement behavior.

Technologies:

RandomForestRegressor

OneHotEncoder

Feature scaling

Train-test evaluation

Outputs:

quiz_score_predictions.csv

Power BI performance dashboard

Dashboard Preview:


 Engagement Classification (High / Medium / Low)

A RandomForestClassifier predicts the engagement category.

Output:

engagement_level_predictions.csv

Agent 2 — Recommendation Agent

Uses engagement patterns + clustering to provide personalized study strategies.

✔ Steps:

K-Means clustering

Risk segmentation

AI-generated recommendations

✔ Output File:

student_recommendations.csv

Recommendation Dashboard:


Agent 3 — Anomaly Detection Agent

Detects students with unusual behavior to predict potential dropouts.

✔ Model:

Isolation Forest

Scaled numeric features

5% contamination

✔ Outputs:

student_anomalies.csv

dropout_risk_flag (High Risk / Normal)

 Anomalies Dashboard:


Project Structure
Students_eng_ai/
│
├─ data/
│   ├─ raw/
│   └─ processed/
│       ├─ quiz_score_predictions.csv
│       ├─ engagement_level_predictions.csv
│       ├─ student_recommendations.csv
│       └─ student_anomalies.csv
│
├─ models/
│   ├─ quiz_score_regressor.joblib
│   └─ engagement_level_classifier.joblib
│
├─ src/
│   ├─ 01_load_and_inspect.py
│   ├─ 02_train_regression_model.py
│   ├─ 03_train_engagement_classifier.py
│   ├─ 04_recommendation_agent.py
│   └─ 05_anomaly_detection_agent.py
│
├─ reports/
│   └─ powerbi/
│       ├─ Student_Performance.pbix
│       ├─ Recommendations.pbix
│       └─ Anomalies.pbix



Technologies Used
Python

Pandas

NumPy

Scikit-Learn

Joblib

K-Means Clustering

Isolation Forest

 Power BI

Predictive dashboards

Risk segmentation

Recommendation tables

Anomaly visualizations
 Tools

VS Code

Git & GitHub

Jupyter (optional)
