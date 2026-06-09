import os
import warnings
import subprocess

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA,
    PROCESSED_DATA,
    METRICS_OUTPUT,
    SCORED_OUTPUT,
    SUSPICIOUS_OUTPUT,
)

from src.load_data import load_sysmon_logs
from src.preprocess import preprocess_logs

from src.train_models import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_neural_network,
    train_isolation_forest,
    train_one_class_svm,
    train_lof,
    train_elliptic_envelope,
    tune_random_forest,
)

from src.evaluate import (
    evaluate_classifier,
    evaluate_predictions,
    find_best_threshold,
    get_confusion_matrix,
)

warnings.filterwarnings("ignore")


def anomaly_to_binary(predictions):
    return [1 if pred == -1 else 0 for pred in predictions]


def launch_dashboard():
    import os
    import sys
    import subprocess

    project_dir = os.path.dirname(os.path.abspath(__file__))

    dashboard_file = os.path.join(
        project_dir,
        "dashboard.py"
    )

    if not os.path.exists(dashboard_file):
        print(f"Dashboard file not found: {dashboard_file}")
        return

    print("Launching Streamlit dashboard...")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_file,
        ],
        cwd=project_dir,
    )


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")

    df = load_sysmon_logs(RAW_DATA)

    print(f"Loaded {len(df)} events")

    print("Preprocessing...")

    df, features = preprocess_logs(df)

    if "Target" not in df.columns:
        df["Target"] = df["RuleRiskScore"].apply(
            lambda score: 1 if score >= 3 else 0
        )

        print("Created weak rule-based Target from RuleRiskScore.")
        print("For better accuracy, replace this with real labels in your CSV.")

    print("Features used by the models:")
    print(features)

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
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_metrics = []

    print("Training Logistic Regression...")
    log_model = train_logistic_regression(X_train_scaled, y_train)
    log_threshold, _ = find_best_threshold(log_model, X_test_scaled, y_test)

    all_metrics.append(
        evaluate_classifier(
            log_model,
            X_test_scaled,
            y_test,
            "Logistic Regression",
            threshold=log_threshold,
        )
    )

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_threshold, _ = find_best_threshold(rf_model, X_test, y_test)

    all_metrics.append(
        evaluate_classifier(
            rf_model,
            X_test,
            y_test,
            "Random Forest",
            threshold=rf_threshold,
        )
    )

    print("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_threshold, _ = find_best_threshold(gb_model, X_test, y_test)

    all_metrics.append(
        evaluate_classifier(
            gb_model,
            X_test,
            y_test,
            "Gradient Boosting",
            threshold=gb_threshold,
        )
    )

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    if xgb_model is not None:
        xgb_threshold, _ = find_best_threshold(xgb_model, X_test, y_test)

        all_metrics.append(
            evaluate_classifier(
                xgb_model,
                X_test,
                y_test,
                "XGBoost",
                threshold=xgb_threshold,
            )
        )

    print("Training Neural Network...")
    nn_model = train_neural_network(X_train_scaled, y_train)
    nn_threshold, _ = find_best_threshold(nn_model, X_test_scaled, y_test)

    all_metrics.append(
        evaluate_classifier(
            nn_model,
            X_test_scaled,
            y_test,
            "Neural Network",
            threshold=nn_threshold,
        )
    )

    print("Training Isolation Forest...")
    iso_model = train_isolation_forest(X_train_scaled)
    iso_preds = anomaly_to_binary(iso_model.predict(X_test_scaled))

    all_metrics.append(
        evaluate_predictions(
            y_test,
            iso_preds,
            "Isolation Forest",
        )
    )

    print("Training One-Class SVM...")
    ocsvm_model = train_one_class_svm(X_train_scaled)
    ocsvm_preds = anomaly_to_binary(ocsvm_model.predict(X_test_scaled))

    all_metrics.append(
        evaluate_predictions(
            y_test,
            ocsvm_preds,
            "One-Class SVM",
        )
    )

    print("Training Local Outlier Factor...")
    lof_model = train_lof(X_train_scaled)
    lof_preds = anomaly_to_binary(lof_model.predict(X_test_scaled))

    all_metrics.append(
        evaluate_predictions(
            y_test,
            lof_preds,
            "Local Outlier Factor",
        )
    )

    print("Training Elliptic Envelope...")
    ee_model = train_elliptic_envelope(X_train_scaled)
    ee_preds = anomaly_to_binary(ee_model.predict(X_test_scaled))

    all_metrics.append(
        evaluate_predictions(
            y_test,
            ee_preds,
            "Elliptic Envelope",
        )
    )

    print("Running hyperparameter tuning for Random Forest...")

    best_rf_model, best_params, best_cv_score = tune_random_forest(
        X_train,
        y_train,
    )

    best_rf_threshold, _ = find_best_threshold(
        best_rf_model,
        X_test,
        y_test,
        metric="f1",
    )

    all_metrics.append(
        evaluate_classifier(
            best_rf_model,
            X_test,
            y_test,
            "Tuned Random Forest",
            threshold=best_rf_threshold,
        )
    )

    print("Best Random Forest parameters:")
    print(best_params)

    print("Best CV F1 score:")
    print(best_cv_score)

    print("Best Random Forest threshold:")
    print(best_rf_threshold)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(
        by="f1",
        ascending=False,
    )

    print("Model comparison:")
    print(metrics_df)

    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"Saved -> {METRICS_OUTPUT}")

    print("Creating confusion matrix...")

    best_rf_scores = best_rf_model.predict_proba(X_test)[:, 1]
    best_rf_preds = (best_rf_scores >= best_rf_threshold).astype(int)

    cm = get_confusion_matrix(y_test, best_rf_preds)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Threat"],
        columns=["Predicted Normal", "Predicted Threat"],
    )

    confusion_matrix_output = "output/confusion_matrix.csv"
    cm_df.to_csv(confusion_matrix_output)

    print(f"Saved -> {confusion_matrix_output}")

    print("Scoring all logs with best model...")

    all_scores = best_rf_model.predict_proba(X)[:, 1]

    df["ThreatProbability"] = all_scores
    df["Prediction"] = (all_scores >= best_rf_threshold).astype(int)

    df.to_csv(SCORED_OUTPUT, index=False)
    print(f"Saved -> {SCORED_OUTPUT}")

    suspicious_df = df[df["Prediction"] == 1]
    suspicious_df.to_csv(SUSPICIOUS_OUTPUT, index=False)

    print(f"Saved -> {SUSPICIOUS_OUTPUT}")
    print(f"Found {len(suspicious_df)} suspicious events")

    print("Pipeline complete.")

    launch_dashboard()


if __name__ == "__main__":
    main()