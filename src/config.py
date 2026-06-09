# Här samlas alla filvägar som används i projektet.
# Fördelen är att man bara behöver ändra sökvägen på ett ställe.

# Rådata från Sysmon-exporten
RAW_DATA = "data/raw/sysmon_logs.csv"

# Förbehandlad data
PROCESSED_DATA = "data/processed/processed_logs.csv"

# Resultat från modelljämförelsen
METRICS_OUTPUT = "output/model_metrics.csv"

# Alla loggar med AI-modellens prediktioner
SCORED_OUTPUT = "output/scored_logs.csv"

# Endast händelser som modellen klassar som misstänkta
SUSPICIOUS_OUTPUT = "output/suspicious_events.csv"