# %%
import pandas as pd

# %%
def load_sysmon_logs(filepath):
    """
    Läs in Sysmon-loggar från CSV
    """

    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} events")

    return df