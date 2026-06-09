import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_predictions(y_test, y_pred, model_name="Model", threshold=None):
    """
    Utvärderar en modells prediktioner.

    y_test = riktiga värden
    y_pred = modellens prediktioner
    model_name = modellens namn
    threshold = gränsvärde om modellen använder sannolikheter

    Returnerar accuracy, precision, recall och F1-score.
    """

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Lägger till threshold om det används
    if threshold is not None:
        result["threshold"] = threshold

    return result


def evaluate_classifier(model, X_test, y_test, model_name="Model", threshold=0.5):
    """
    Utvärderar en klassificeringsmodell.

    Om modellen kan ge sannolikheter används predict_proba().
    Annars används vanlig predict().
    """

    # Om modellen kan ge sannolikhet för klass 1
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]

        # Gör om sannolikheter till 0 eller 1
        y_pred = (y_scores >= threshold).astype(int)

    else:
        # För modeller som inte stödjer predict_proba
        y_pred = model.predict(X_test)

    return evaluate_predictions(
        y_test,
        y_pred,
        model_name,
        threshold=threshold,
    )


def find_best_threshold(model, X_valid, y_valid, metric="f1"):
    """
    Testar flera olika threshold-värden och väljer det bästa.

    Threshold betyder gränsen för när en sannolikhet ska räknas som hot.

    Exempel:
    Om threshold = 0.60 krävs minst 60% sannolikhet för att klassas som hot.
    """

    # Om modellen inte stödjer sannolikheter används standardvärdet 0.5
    if not hasattr(model, "predict_proba"):
        return 0.5, None

    # Hämtar sannolikheter för klass 1
    y_scores = model.predict_proba(X_valid)[:, 1]

    # Testar thresholds från 0.05 till 0.95
    thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_score = -1

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if metric == "recall":
            score = recall_score(y_valid, y_pred, zero_division=0)
        else:
            score = f1_score(y_valid, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def get_confusion_matrix(y_test, y_pred):
    """
    Skapar en confusion matrix.

    Den visar:
    - true negatives
    - false positives
    - false negatives
    - true positives
    """

    return confusion_matrix(y_test, y_pred)