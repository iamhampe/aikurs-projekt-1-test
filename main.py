import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA,
    PROCESSED_DATA,
    METRICS_OUTPUT,
    SCORED_OUTPUT,
    SUSPICIOUS_OUTPUT
)

from src.load_data import load_sysmon_logs
from src.preprocess import preprocess_logs

from src.train_models import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_isolation_forest,
    tune_random_forest
)

from src.evaluate import (
    evaluate_classifier,
    evaluate_predictions,
    get_confusion_matrix
)


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    df = load_sysmon_logs(RAW_DATA)
    print(f"Loaded {len(df)} events")

    print("Preprocessing...")
    df, features = preprocess_logs(df)

    # =====================================================
    # CREATE TARGET COLUMN FOR SUPERVISED LEARNING
    # 0 = normal event
    # 1 = suspicious/threat event
    # =====================================================
    if "Target" not in df.columns:
        df["Target"] = df["SuspiciousKeywordScore"].apply(
            lambda score: 1 if score > 0 else 0
        )

    print("Target distribution:")
    print(df["Target"].value_counts())

    df.to_csv(PROCESSED_DATA, index=False)
    print(f"Saved -> {PROCESSED_DATA}")

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    all_metrics = []

    print("Training Logistic Regression...")
    log_model = train_logistic_regression(X_train, y_train)
    all_metrics.append(
        evaluate_classifier(log_model, X_test, y_test, "Logistic Regression")
    )

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    all_metrics.append(
        evaluate_classifier(rf_model, X_test, y_test, "Random Forest")
    )

    print("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    all_metrics.append(
        evaluate_classifier(gb_model, X_test, y_test, "Gradient Boosting")
    )

    print("Training Isolation Forest...")
    iso_model = train_isolation_forest(X_train)

    iso_preds = iso_model.predict(X_test)
    iso_preds = [1 if pred == -1 else 0 for pred in iso_preds]

    all_metrics.append(
        evaluate_predictions(y_test, iso_preds, "Isolation Forest")
    )

    print("Running hyperparameter tuning for Random Forest...")
    best_rf_model, best_params, best_cv_score = tune_random_forest(
        X_train,
        y_train
    )

    all_metrics.append(
        evaluate_classifier(
            best_rf_model,
            X_test,
            y_test,
            "Tuned Random Forest"
        )
    )

    print("Best Random Forest parameters:")
    print(best_params)

    print("Best CV F1 score:")
    print(best_cv_score)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(by="f1", ascending=False)

    print("Model comparison:")
    print(metrics_df)

    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"Saved -> {METRICS_OUTPUT}")

    print("Creating confusion matrix...")
    best_rf_preds = best_rf_model.predict(X_test)
    cm = get_confusion_matrix(y_test, best_rf_preds)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Threat"],
        columns=["Predicted Normal", "Predicted Threat"]
    )

    confusion_matrix_output = "output/confusion_matrix.csv"
    cm_df.to_csv(confusion_matrix_output)
    print(f"Saved -> {confusion_matrix_output}")

    print("Scoring all logs with best model...")
    df["Prediction"] = best_rf_model.predict(X)

    df.to_csv(SCORED_OUTPUT, index=False)
    print(f"Saved -> {SCORED_OUTPUT}")

    suspicious_df = df[df["Prediction"] == 1]
    suspicious_df.to_csv(SUSPICIOUS_OUTPUT, index=False)

    print(f"Saved -> {SUSPICIOUS_OUTPUT}")
    print(f"Found {len(suspicious_df)} suspicious events")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()