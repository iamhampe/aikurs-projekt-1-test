import re
import pandas as pd


def extract_time_features(df):
    if "TimeCreated" in df.columns:
        df["TimeCreated"] = pd.to_datetime(df["TimeCreated"], errors="coerce")
        df["Hour"] = df["TimeCreated"].dt.hour.fillna(0).astype(int)
        df["DayOfWeek"] = df["TimeCreated"].dt.dayofweek.fillna(0).astype(int)
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        df["IsNight"] = df["Hour"].between(0, 6).astype(int)
    else:
        df["Hour"] = 0
        df["DayOfWeek"] = 0
        df["IsWeekend"] = 0
        df["IsNight"] = 0
    return df


def contains_any(text, patterns):
    return int(any(pattern in text for pattern in patterns))


def regex_any(text, pattern):
    return int(bool(re.search(pattern, text, re.IGNORECASE)))


def create_security_features(df):
    if "Message" not in df.columns:
        df["Message"] = ""

    msg = df["Message"].fillna("").astype(str).str.lower()

    df["MessageLength"] = msg.str.len()
    df["CommandLineLength"] = msg.apply(lambda x: len(re.findall(r"\S+", x)))

    # PowerShell abuse
    df["HasPowerShell"] = msg.str.contains("powershell", regex=False).astype(int)
    df["HasEncodedCommand"] = msg.str.contains(r"encodedcommand|\s-enc\s|\s-e\s", regex=True).astype(int)
    df["HasBypass"] = msg.str.contains(r"executionpolicy bypass|bypass", regex=True).astype(int)
    df["HasHiddenWindow"] = msg.str.contains(r"windowstyle hidden|-w hidden", regex=True).astype(int)
    df["HasNoProfile"] = msg.str.contains(r"nop|-noprofile", regex=True).astype(int)

    # Download / remote execution
    df["HasDownload"] = msg.str.contains(
        r"downloadstring|downloadfile|invoke-webrequest|iwr|wget|curl|bitsadmin",
        regex=True
    ).astype(int)
    df["HasHttp"] = msg.str.contains(r"http://|https://", regex=True).astype(int)

    # LOLBins
    df["HasLolbin"] = msg.str.contains(
        r"rundll32|regsvr32|certutil|mshta|wmic|bitsadmin|installutil|msbuild|regasm|regsvcs",
        regex=True
    ).astype(int)

    # Script execution
    df["HasScriptHost"] = msg.str.contains(r"wscript|cscript|jscript|vbscript|\.vbs|\.js|\.hta", regex=True).astype(int)

    # Suspicious paths
    df["HasTempPath"] = msg.str.contains(r"\\temp\\|appdata|programdata|public\\|downloads", regex=True).astype(int)
    df["HasStartupPath"] = msg.str.contains(r"startup|start menu\\programs\\startup", regex=True).astype(int)

    # Persistence
    df["HasPersistenceKeyword"] = msg.str.contains(
        r"schtasks|task scheduler|run key|autorun|startup|services\\|new-service|sc.exe create",
        regex=True
    ).astype(int)

    # Credentials / dumping
    df["HasCredentialKeyword"] = msg.str.contains(
        r"mimikatz|lsass|sekurlsa|credential|password|dump|procdump|comsvcs.dll",
        regex=True
    ).astype(int)

    # Discovery commands
    df["HasDiscoveryCommand"] = msg.str.contains(
        r"whoami|ipconfig|systeminfo|net user|net localgroup|nltest|quser|hostname",
        regex=True
    ).astype(int)

    # Defense evasion
    df["HasDefenseEvasion"] = msg.str.contains(
        r"vssadmin delete|bcdedit|disableantispyware|set-mppreference|add-mppreference|wevtutil cl|clear-eventlog",
        regex=True
    ).astype(int)

    # Registry activity
    df["HasRegistryModification"] = msg.str.contains(
        r"registry value set|reg add|reg delete|currentversion\\run|currentversion\\runonce",
        regex=True
    ).astype(int)

    # Network indicators
    df["HasNetworkIndicator"] = msg.str.contains(
        r"destinationip|destinationport|sourceip|sourceport|dns query|tcp|udp",
        regex=True
    ).astype(int)

    # Common suspicious extensions
    df["HasSuspiciousExtension"] = msg.str.contains(
        r"\.ps1|\.bat|\.cmd|\.vbs|\.js|\.hta|\.dll|\.scr|\.exe",
        regex=True
    ).astype(int)

    # Base64-like long strings
    df["HasLongBase64LikeString"] = msg.str.contains(
        r"[a-zA-Z0-9+/]{80,}={0,2}",
        regex=True
    ).astype(int)

    # Parent/child suspicious combos in raw message
    df["HasOfficeSpawningProcess"] = msg.str.contains(
        r"winword\.exe|excel\.exe|powerpnt\.exe|outlook\.exe",
        regex=True
    ).astype(int) & msg.str.contains(
        r"powershell|cmd\.exe|wscript|cscript|mshta|rundll32",
        regex=True
    ).astype(int)

    df["HasBrowserSpawningScript"] = msg.str.contains(
        r"chrome\.exe|msedge\.exe|firefox\.exe",
        regex=True
    ).astype(int) & msg.str.contains(
        r"powershell|cmd\.exe|wscript|cscript|mshta",
        regex=True
    ).astype(int)

    # Suspicious score used only for weak target creation
    suspicious_keywords = [
        "powershell", "encodedcommand", "-enc", "rundll32", "regsvr32",
        "certutil", "bitsadmin", "wscript", "cscript", "mshta", "appdata",
        "startup", "schtasks", "mimikatz", "lsass", "vssadmin", "bcdedit",
        "whoami", "net user", "downloadstring", "invoke-webrequest",
        "wevtutil cl", "procdump"
    ]

    df["SuspiciousKeywordScore"] = msg.apply(
        lambda text: sum(1 for word in suspicious_keywords if word in text)
    )

    # Stronger risk score for analysis
    risk_columns = [
        "HasPowerShell",
        "HasEncodedCommand",
        "HasBypass",
        "HasHiddenWindow",
        "HasNoProfile",
        "HasDownload",
        "HasHttp",
        "HasLolbin",
        "HasScriptHost",
        "HasTempPath",
        "HasStartupPath",
        "HasPersistenceKeyword",
        "HasCredentialKeyword",
        "HasDiscoveryCommand",
        "HasDefenseEvasion",
        "HasRegistryModification",
        "HasNetworkIndicator",
        "HasSuspiciousExtension",
        "HasLongBase64LikeString",
        "HasOfficeSpawningProcess",
        "HasBrowserSpawningScript",
        "IsNight",
        "IsWeekend",
    ]

    df["RuleRiskScore"] = df[risk_columns].sum(axis=1)

    return df


def add_optional_numeric_features(df):
    optional_numeric_columns = [
        "Id",
        "EventID",
        "ProcessId",
        "ParentProcessId",
        "SourcePort",
        "DestinationPort"
    ]

    features = []

    for column in optional_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
            features.append(column)

    return df, features


def preprocess_logs(df):
    df = extract_time_features(df)
    df = create_security_features(df)
    df, optional_numeric_features = add_optional_numeric_features(df)

    features = optional_numeric_features + [
        "Hour",
        "DayOfWeek",
        "IsWeekend",
        "IsNight",
        "MessageLength",
        "CommandLineLength",
        "HasPowerShell",
        "HasEncodedCommand",
        "HasBypass",
        "HasHiddenWindow",
        "HasNoProfile",
        "HasDownload",
        "HasHttp",
        "HasLolbin",
        "HasScriptHost",
        "HasTempPath",
        "HasStartupPath",
        "HasPersistenceKeyword",
        "HasCredentialKeyword",
        "HasDiscoveryCommand",
        "HasDefenseEvasion",
        "HasRegistryModification",
        "HasNetworkIndicator",
        "HasSuspiciousExtension",
        "HasLongBase64LikeString",
        "HasOfficeSpawningProcess",
        "HasBrowserSpawningScript",
    ]

    features = [feature for feature in features if feature in df.columns]

    return df, features