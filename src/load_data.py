import pandas as pd


def load_sysmon_logs(filepath):
    """
    Läser in Sysmon-loggar från en CSV-fil.

    filepath = sökvägen till CSV-filen

    Funktionen returnerar en pandas DataFrame.
    """

    # Läser CSV-filen
    df = pd.read_csv(filepath)

    # Skriver ut hur många händelser som laddades
    print(f"Loaded {len(df)} events")

    return df