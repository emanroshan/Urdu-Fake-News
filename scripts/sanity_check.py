import pandas as pd
from pathlib import Path

FILE = Path("/content/urdu-fake-news/data/Combined.cleaned.csv")

df = pd.read_csv(FILE)

assert len(df) >= 9000, f"ERROR: Expected 10083 rows, got {len(df)}"
print("Row count OK")

labels = set(df["label"].unique())
assert labels == {"True", "Fake"}, f"ERROR: Unexpected labels {labels}"
print("Label values OK")

assert df["text"].isna().sum() == 0, "ERROR: Missing values in text column!"
print("No missing text")

