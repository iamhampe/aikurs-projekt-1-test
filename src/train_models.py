from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier

# Försöker importera XGBoost.
# Om XGBoost inte är installerat hoppar programmet över den modellen.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Samma random_state gör att resultaten blir mer reproducerbara
RANDOM_STATE = 42


def train_logistic_regression(X, y):
    """
    Tränar Logistic Regression.

    Bra enkel modell som fungerar som baseline.
    class_weight='balanced' hjälper om det finns fler normala än misstänkta loggar.
    """

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    return model


def train_random_forest(X, y):
    """
    Tränar Random Forest.

    Random Forest består av många beslutsträd.
    Den fungerar ofta bra på tabulär data.
    """

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X, y)

    return model


def train_gradient_boosting(X, y):
    """
    Tränar Gradient Boosting.

    Modellen bygger flera svaga träd efter varandra
    och försöker förbättra tidigare fel.
    """

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    return model


def train_xgboost(X, y):
    """
    Tränar XGBoost om paketet är installerat.

    XGBoost är en kraftfull boosting-modell.
    """

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed.")
        print("Skipping XGBoost.")
        return None

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X, y)

    return model


def train_neural_network(X, y):
    """
    Tränar ett neuralt nätverk.

    hidden_layer_sizes=(64, 32) betyder:
    - första dolda lagret har 64 neuroner
    - andra dolda lagret har 32 neuroner
    """

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    return model


def train_isolation_forest(X):
    """
    Tränar Isolation Forest.

    Detta är en anomaly detection-modell.
    Den försöker hitta datapunkter som sticker ut från resten.
    """

    model = IsolationForest(
        n_estimators=400,
        contamination=0.10,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X)

    return model


def train_one_class_svm(X):
    """
    Tränar One-Class SVM.

    Modellen lär sig vad som verkar normalt
    och markerar avvikande händelser som misstänkta.
    """

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    model.fit(X)

    return model


def train_lof(X):
    """
    Tränar Local Outlier Factor.

    LOF jämför datapunkter med sina närmaste grannar
    och hittar punkter som verkar ovanliga lokalt.
    """

    model = LocalOutlierFactor(
        n_neighbors=10,
        contamination=0.10,
        novelty=True,
    )

    model.fit(X)

    return model


def train_elliptic_envelope(X):
    """
    Tränar Elliptic Envelope.

    Modellen antar ungefär normalfördelad data
    och markerar punkter långt från centrum som avvikelser.
    """

    model = EllipticEnvelope(
        contamination=0.10,
        random_state=RANDOM_STATE,
    )

    model.fit(X)

    return model


def tune_random_forest(X, y):
    """
    Testar olika inställningar för Random Forest med GridSearchCV.

    Den väljer kombinationen som ger bäst F1-score.
    """

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    grid_search.fit(X, y)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )