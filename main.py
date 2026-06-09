import os
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importerar sökvägar till filer från config.py
from src.config import (
    RAW_DATA,
    PROCESSED_DATA,
    METRICS_OUTPUT,
    SCORED_OUTPUT,
    SUSPICIOUS_OUTPUT,
)

# Importerar funktion för att läsa in Sysmon-loggar
from src.load_data import load_sysmon_logs

# Importerar funktion som rensar data och skapar features
from src.preprocess import preprocess_logs

# Importerar alla modeller som ska tränas
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

# Importerar funktioner för utvärdering av modeller
from src.evaluate import (
    evaluate_classifier,
    evaluate_predictions,
    find_best_threshold,
    get_confusion_matrix,
)

# Döljer varningar så utskriften blir renare
warnings.filterwarnings("ignore")


def anomaly_to_binary(predictions):
    """
    Gör om anomaly detection-resultat till 0 och 1.

    Vissa modeller returnerar:
    -1 = avvikelse / misstänkt
     1 = normal

    Vi gör om detta till:
    1 = misstänkt
    0 = normal
    """
    return [1 if pred == -1 else 0 for pred in predictions]


def main():
    """
    Huvudfunktionen som kör hela projektets pipeline:
    1. Skapar mappar
    2. Läser in data
    3. Förbehandlar data
    4. Skapar target om den saknas
    5. Tränar flera AI-modeller
    6. Jämför modeller
    7. Sparar resultat till CSV-filer
    """

    # Skapar mappar om de inte redan finns
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")

    # Läser in Sysmon-loggar från CSV-filen
    df = load_sysmon_logs(RAW_DATA)

    print(f"Loaded {len(df)} events")

    print("Preprocessing...")

    # Rensar data och skapar numeriska features som modellerna kan använda
    df, features = preprocess_logs(df)

    # Om datasetet inte redan har en Target-kolumn skapas en enkel regelbaserad target
    # Target = 1 betyder misstänkt händelse
    # Target = 0 betyder normal händelse
    if "Target" not in df.columns:
        df["Target"] = df["RuleRiskScore"].apply(
            lambda score: 1 if score >= 3 else 0
        )

        print("Created weak rule-based Target from RuleRiskScore.")
        print("For better accuracy, replace this with real labels in your CSV.")

    # Visar vilka features som används av modellerna
    print("Features used by the models:")
    print(features)

    # Visar hur många normala och misstänkta händelser som finns
    print("Target distribution:")
    print(df["Target"].value_counts())

    # Sparar den behandlade datan
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"Saved -> {PROCESSED_DATA}")

    # X innehåller input-features
    # y innehåller facit/target
    X = df[features]
    y = df["Target"]

    # Delar upp datan i träningsdata och testdata
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # StandardScaler gör att numeriska värden hamnar på liknande skala
    # Detta är viktigt för t.ex. Logistic Regression, Neural Network och SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    # Lista där alla modellresultat sparas
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

    # XGBoost körs bara om paketet är installerat
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
        evaluate_predictions(y_test, iso_preds, "Isolation Forest")
    )

    print("Training One-Class SVM...")
    ocsvm_model = train_one_class_svm(X_train_scaled)
    ocsvm_preds = anomaly_to_binary(ocsvm_model.predict(X_test_scaled))
    all_metrics.append(
        evaluate_predictions(y_test, ocsvm_preds, "One-Class SVM")
    )

    print("Training Local Outlier Factor...")
    lof_model = train_lof(X_train_scaled)
    lof_preds = anomaly_to_binary(lof_model.predict(X_test_scaled))
    all_metrics.append(
        evaluate_predictions(y_test, lof_preds, "Local Outlier Factor")
    )

    print("Training Elliptic Envelope...")
    ee_model = train_elliptic_envelope(X_train_scaled)
    ee_preds = anomaly_to_binary(ee_model.predict(X_test_scaled))
    all_metrics.append(
        evaluate_predictions(y_test, ee_preds, "Elliptic Envelope")
    )

    print("Running hyperparameter tuning for Random Forest...")

    # Testar flera inställningar för Random Forest och väljer den bästa
    best_rf_model, best_params, best_cv_score = tune_random_forest(
        X_train,
        y_train,
    )

    # Hittar bästa sannolikhetsgräns för modellen
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

    # Gör om modellresultaten till en tabell
    metrics_df = pd.DataFrame(all_metrics)

    # Sorterar modellerna efter bäst F1-score
    metrics_df = metrics_df.sort_values(by="f1", ascending=False)

    print("Model comparison:")
    print(metrics_df)

    # Sparar modellernas resultat
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"Saved -> {METRICS_OUTPUT}")

    print("Creating confusion matrix...")

    # Räknar ut sannolikheter för testdatan med bästa modellen
    best_rf_scores = best_rf_model.predict_proba(X_test)[:, 1]

    # Gör om sannolikheter till 0 eller 1
    best_rf_preds = (best_rf_scores >= best_rf_threshold).astype(int)

    # Skapar confusion matrix
    cm = get_confusion_matrix(y_test, best_rf_preds)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Threat"],
        columns=["Predicted Normal", "Predicted Threat"],
    )

    # Sparar confusion matrix
    confusion_matrix_output = "output/confusion_matrix.csv"
    cm_df.to_csv(confusion_matrix_output)

    print(f"Saved -> {confusion_matrix_output}")

    print("Scoring all logs with best model...")

    # Kör bästa modellen på alla loggar
    all_scores = best_rf_model.predict_proba(X)[:, 1]

    # Lägger till sannolikhet för hot i dataframe
    df["ThreatProbability"] = all_scores

    # Lägger till slutlig prediktion
    df["Prediction"] = (all_scores >= best_rf_threshold).astype(int)

    # Sparar alla loggar med prediktioner
    df.to_csv(SCORED_OUTPUT, index=False)
    print(f"Saved -> {SCORED_OUTPUT}")

    # Filtrerar ut endast misstänkta händelser
    suspicious_df = df[df["Prediction"] == 1]

    # Sparar misstänkta händelser
    suspicious_df.to_csv(SUSPICIOUS_OUTPUT, index=False)
    print(f"Saved -> {SUSPICIOUS_OUTPUT}")

    print(f"Found {len(suspicious_df)} suspicious events")
    print("Pipeline complete.")


# Kör main() endast om filen startas direkt
if __name__ == "__main__":
    main()