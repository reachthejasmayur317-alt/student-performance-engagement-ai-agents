import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# -------- Paths --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "student_cleaned.csv"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PREDICTIONS_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "quiz_score_predictions.csv"
)

MODEL_PATH = os.path.join(MODELS_DIR, "quiz_score_regressor.joblib")


def load_data():
    print(f" Loading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(" Data loaded. Shape:", df.shape)
    return df


def build_and_train_model(df: pd.DataFrame):
    # ---- 1. Define target and features ----
    target_col = "quiz_score_avg"

    # All columns except target
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
        "engagement_level",  # can also be useful as input
    ]

    # Filter to keep only existing columns (safety)
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ---- 2. Identify numeric & categorical columns ----
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\n Numeric features:", numeric_features)
    print(" Categorical features:", categorical_features)

    # ---- 3. Preprocessing: OneHotEncode categoricals ----
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ---- 4. Define the model ----
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # ---- 5. Build pipeline ----
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # ---- 6. Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        # 80% train, 20% test
    )

    print("\n Training samples:", X_train.shape[0])
    print(" Test samples:", X_test.shape[0])

    # ---- 7. Train the model ----
    print("\n Training RandomForestRegressor...")
    model.fit(X_train, y_train)
    print(" Model training complete.")

    # ---- 8. Evaluate ----
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n Evaluation Metrics (Agent 1 - Regression):")
    print(f"  MAE  (Mean Absolute Error): {mae:.3f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"  R²   (R-squared): {r2:.3f}")

    # ---- 9. Save predictions for Power BI / analysis ----
    results_df = X_test.copy()
    results_df["quiz_score_actual"] = y_test.values
    results_df["quiz_score_predicted"] = y_pred

    results_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n Predictions saved to: {PREDICTIONS_PATH}")

    # ---- 10. Save model ----
    joblib.dump(model, MODEL_PATH)
    print(f" Trained model saved to: {MODEL_PATH}")

    return model, results_df


def main():
    df = load_data()
    build_and_train_model(df)


if __name__ == "__main__":
    main()
