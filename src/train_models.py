# %%
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

# %%
def train_random_forest(X, y):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


# %%
def train_isolation_forest(X):

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    return model