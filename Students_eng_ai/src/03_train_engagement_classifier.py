import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# -------- Paths --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_cleaned.csv"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PREDICTIONS_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "engagement_level_predictions.csv"
)

MODEL_PATH = os.path.join(MODELS_DIR, "engagement_level_classifier.joblib")


def load_data():
    print(f"📥 Loading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print("✅ Data loaded. Shape:", df.shape)
    return df


def build_and_train_classifier(df: pd.DataFrame):
    target_col = "engagement_level"

    feature_cols = [
        "time_spent_weekly",
        "forum_posts",
        "video_watched_percent",
        "assignments_submitted",
        "login_frequency",
        "session_duration_avg",
        "device_type",
        "course_difficulty",
        "region",
        "quiz_score_avg",
    ]

    # keep only columns that exist (safety)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ----- Encode target labels -----
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\n🔤 Engagement level classes:", list(label_encoder.classes_))

    # ----- Separate numeric & categorical features -----
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\n📊 Numeric features:", numeric_features)
    print("🔤 Categorical features:", categorical_features)

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("\n🚂 Training samples:", X_train.shape[0])
    print("🧪 Test samples:", X_test.shape[0])

    print("\n🧠 Training RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("✅ Classifier training complete.")

    # ----- Evaluation -----
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📈 Accuracy: {acc:.3f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ----- Save predictions for Power BI -----
    X_test_reset = X_test.reset_index(drop=True)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    results_df = X_test_reset.copy()
    results_df["engagement_actual"] = y_test_labels
    results_df["engagement_predicted"] = y_pred_labels

    results_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n💾 Engagement predictions saved to: {PREDICTIONS_PATH}")

    # ----- Save model & label encoder -----
    joblib.dump(
        {"model": model, "label_encoder": label_encoder},
        MODEL_PATH
    )
    print(f"💾 Trained classifier + label encoder saved to: {MODEL_PATH}")

    return model, results_df


def main():
    df = load_data()
    build_and_train_classifier(df)


if __name__ == "__main__":
    main()
