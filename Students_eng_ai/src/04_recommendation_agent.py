import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_recommendations.csv"
)


def load_data():
    print(f"Loading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print("Data loaded. Shape:", df.shape)
    return df


def build_clusters(df: pd.DataFrame, n_clusters: int = 4):
    """
    Cluster students based on engagement behaviour.
    """
    numeric_features = [
        "time_spent_weekly",
        "quiz_score_avg",
        "forum_posts",
        "video_watched_percent",
        "assignments_submitted",
        "login_frequency",
        "session_duration_avg",
    ]

    # keep only columns that actually exist
    numeric_features = [c for c in numeric_features if c in df.columns]

    X = df[numeric_features].copy()

    # handle any missing values just in case
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    print("\nFitting KMeans clustering...")
    cluster_labels = kmeans.fit_predict(X_scaled)
    print("Clustering complete.")

    df["engagement_cluster"] = cluster_labels

    # cluster profile summary
    cluster_summary = df.groupby("engagement_cluster")[numeric_features].mean()
    print("\n Cluster behaviour summary (means):")
    print(cluster_summary.round(2))

    return df, cluster_summary


def assign_risk_segment(row):
    """
    Simple rule-based risk segmentation based on quiz score and engagement level.
    """
    score = row.get("quiz_score_avg", np.nan)
    level = row.get("engagement_level", "")

    if pd.isna(score):
        score = 0

    # high risk
    if score < 60 or level == "Low":
        return "High Risk"

    # medium risk
    if 60 <= score < 80 or level == "Medium":
        return "Medium Risk"

    # otherwise low risk
    return "Low Risk"


def generate_recommendation(row):
    """
    Generate a text recommendation for a single student row.
    You can tweak the messages to sound more polished.
    """
    score = row.get("quiz_score_avg", 0)
    level = row.get("engagement_level", "Unknown")
    cluster = row.get("engagement_cluster", -1)
    risk = row.get("risk_segment", "Unknown")

    time_spent = row.get("time_spent_weekly", 0)
    videos = row.get("video_watched_percent", 0)
    assigns = row.get("assignments_submitted", 0)
    logins = row.get("login_frequency", 0)

    # Base template
    rec_parts = []

    rec_parts.append(
        f"Current status: {risk} student with {level} engagement and average quiz score of {score:.1f}."
    )

    # Study time recommendation
    if time_spent < 5:
        rec_parts.append(
            "Increase weekly study time to at least 5–7 hours and create a fixed daily study slot."
        )
    elif time_spent < 10:
        rec_parts.append(
            "You are studying a moderate amount. Try adding one focused 45–60 minute session on weak topics."
        )
    else:
        rec_parts.append(
            "Your study time is good. Focus on improving quality with spaced revision and practice tests."
        )

    # Video watching behaviour
    if videos < 70:
        rec_parts.append(
            "Your video completion rate is low. Try to finish full lectures and take short notes while watching."
        )
    elif videos < 90:
        rec_parts.append(
            "You watch most videos. Re-watch difficult segments at 1.25x speed and summarize key formulas/concepts."
        )
    else:
        rec_parts.append(
            "Excellent video engagement. Shift more time towards active recall and practice quizzes."
        )

    # Assignments & logins
    if assigns < 5:
        rec_parts.append(
            "Complete all upcoming assignments on time. Use them as checkpoints to test understanding."
        )

    if logins < 3:
        rec_parts.append(
            "Log in more frequently (at least 3–4 times a week) to stay consistent instead of last-minute studying."
        )

    # Cluster-based flavour text
    cluster_msg = {
        0: "You behave like a 'low-activity' learner. Start with small daily goals and track your progress.",
        1: "You are a 'steady' learner. Slightly increasing practice questions can boost your performance.",
        2: "You show 'high-effort' behaviour. Focus on exam strategy and timing to convert effort into marks.",
        3: "Your pattern is mixed. Identify 1–2 core weaknesses (like theory, numericals, or revision) and target them first.",
    }

    if cluster in cluster_msg:
        rec_parts.append(cluster_msg[cluster])

    # Final recommendation text
    return " ".join(rec_parts)


def build_recommendations(df: pd.DataFrame, cluster_summary: pd.DataFrame):
    """
    Add risk segment + recommendation text columns.
    """
    df["risk_segment"] = df.apply(assign_risk_segment, axis=1)
    df["recommendation_text"] = df.apply(generate_recommendation, axis=1)

    return df


def main():
    df = load_data()
    df_with_clusters, cluster_summary = build_clusters(df, n_clusters=4)
    df_final = build_recommendations(df_with_clusters, cluster_summary)

    # Save output
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Student recommendations saved to: {OUTPUT_PATH}")

    # Show a sample
    print("\n Sample recommendations:")
    print(
        df_final[
            [
                "quiz_score_avg",
                "engagement_level",
                "risk_segment",
                "engagement_cluster",
                "recommendation_text",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
