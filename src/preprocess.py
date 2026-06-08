# %%
import pandas as pd

# %%
def extract_time_features(df):

    df["TimeCreated"] = pd.to_datetime(df["TimeCreated"])

    df["Hour"] = df["TimeCreated"].dt.hour
    df["DayOfWeek"] = df["TimeCreated"].dt.dayofweek

    return df


# %%
def create_security_features(df):

    keywords = [
        "powershell",
        "encodedcommand",
        "rundll32",
        "regsvr32",
        "temp",
        "appdata"
    ]

    def suspicious_score(message):

        if pd.isna(message):
            return 0

        msg = str(message).lower()

        score = 0

        for word in keywords:
            if word in msg:
                score += 1

        return score

    df["SuspiciousKeywordScore"] = (
        df["Message"].apply(suspicious_score)
    )

    return df


# %%
def preprocess_logs(df):

    df = extract_time_features(df)

    df = create_security_features(df)

    features = [
        "Id",
        "Hour",
        "DayOfWeek",
        "SuspiciousKeywordScore"
    ]

    return df, features