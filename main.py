# %%
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.config import *

from src.load_data import (
    load_sysmon_logs
)

from src.preprocess import (
    preprocess_logs
)

from src.train_models import (
    train_random_forest,
    train_isolation_forest
)

from src.evaluate import (
    evaluate_classifier
)

from src.export_powerbi import (
    export_scored_logs,
    export_metrics
)

# %%
print("Loading data...")

df = load_sysmon_logs(
    RAW_DATA
)

# %%
print("Preprocessing...")

df, features = preprocess_logs(df)

# %%
"""
Labeling

För projektet använder vi
SuspiciousKeywordScore > 0
som en enkel label.
"""

df["Target"] = (
    df["SuspiciousKeywordScore"] > 0
).astype(int)

# %%
X = df[features]

y = df["Target"]

# %%
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# %%
print("Training Random Forest...")

rf_model = train_random_forest(
    X_train,
    y_train
)

# %%
print("Evaluating Random Forest...")

metrics = evaluate_classifier(
    rf_model,
    X_test,
    y_test
)

print(metrics)

# %%
print("Training Isolation Forest...")

iso_model = train_isolation_forest(
    X
)

# %%
df["AnomalyPrediction"] = (
    iso_model.predict(X)
)

"""
Isolation Forest:

-1 = anomalous

 1 = normal
"""

# %%
export_metrics(
    metrics,
    METRICS_OUTPUT
)

export_scored_logs(
    df,
    SCORED_OUTPUT
)

# %%
suspicious_events = df[
    df["AnomalyPrediction"] == -1
]

suspicious_events.to_csv(
    SUSPICIOUS_OUTPUT,
    index=False
)

print(
    f"Found {len(suspicious_events)} suspicious events"
)

print("Pipeline complete.")