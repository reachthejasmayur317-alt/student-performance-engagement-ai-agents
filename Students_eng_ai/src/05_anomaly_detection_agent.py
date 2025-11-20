import os
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------- PATHS ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_anomalies.csv"
)


def load_data():
    print(f" Loading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(" Data loaded. Shape:", df.shape)
    return df


def detect_anomalies(df: pd.DataFrame):
    # Features used for anomaly detection
    anomaly_features = [
        "time_spent_weekly",
        "forum_posts",
        "video_watched_percent",
        "assignments_submitted",
        "login_frequency",
        "session_duration_avg",
        "quiz_score_avg",
    ]

    anomaly_features = [c for c in anomaly_features if c in df.columns]

    X = df[anomaly_features].copy()

    # Replace missing values
    X = X.fillna(X.median(numeric_only=True))

    # Scale values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n Running Isolation Forest for anomaly detection...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # 5% anomalies
        random_state=42
    )

    iso.fit(X_scaled)

    df["anomaly_score"] = iso.decision_function(X_scaled)
    df["is_outlier"] = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal

    # Convert -1/1 to 1/0 for easier use
    df["is_outlier"] = df["is_outlier"].apply(lambda x: 1 if x == -1 else 0)

    # Create risk flag
    df["dropout_risk_flag"] = df["is_outlier"].apply(
        lambda x: "High Risk" if x == 1 else "Normal"
    )

    print(" Anomalies detected.")
    print(df["dropout_risk_flag"].value_counts())

    return df


def main():
    df = load_data()
    df_anomaly = detect_anomalies(df)

    df_anomaly.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Anomaly file saved to: {OUTPUT_PATH}")

    print("\n Sample anomaly rows:")
    print(
        df_anomaly[
            [
                "quiz_score_avg",
                "engagement_level",
                "anomaly_score",
                "is_outlier",
                "dropout_risk_flag",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
