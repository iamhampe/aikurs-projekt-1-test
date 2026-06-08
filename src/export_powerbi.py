# %%
import pandas as pd

# %%
def export_scored_logs(df, path):

    df.to_csv(path, index=False)

    print(f"Saved -> {path}")


# %%
def export_metrics(metrics, path):

    pd.DataFrame([metrics]).to_csv(
        path,
        index=False
    )

    print(f"Saved -> {path}")