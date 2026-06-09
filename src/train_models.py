from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42


def train_logistic_regression(X, y):
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def train_random_forest(X, y):
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
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost.")
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
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10
    )
    model.fit(X)
    return model


def train_lof(X):
    model = LocalOutlierFactor(
        n_neighbors=10,
        contamination=0.10,
        novelty=True
    )
    model.fit(X)
    return model


def train_elliptic_envelope(X):
    model = EllipticEnvelope(
        contamination=0.10,
        random_state=RANDOM_STATE
    )
    model.fit(X)
    return model


def tune_random_forest(X, y):
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_