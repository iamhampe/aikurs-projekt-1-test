import pandas as pd


def export_scored_logs(df, path):
    """
    Exporterar alla loggar med modellens prediktioner.

    Denna fil kan sedan importeras i Power BI.
    """

    df.to_csv(path, index=False)

    print(f"Saved -> {path}")


def export_metrics(metrics, path):
    """
    Exporterar modellens resultat till CSV.

    metrics kan vara en dictionary med t.ex.
    accuracy, precision, recall och F1-score.
    """

    pd.DataFrame([metrics]).to_csv(
        path,
        index=False,
    )

    print(f"Saved -> {path}")