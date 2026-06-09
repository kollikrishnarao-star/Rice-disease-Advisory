import pandas as pd
import numpy as np
from google.colab import files

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = "pest_5lag_dataset_fixed_log_ver2.csv"
MIN_ACTIVE = 2


# -----------------------------
# UPLOAD CSV FROM USER
# -----------------------------
print("Upload training dataset CSV")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)
df.columns = df.columns.str.strip()


# -----------------------------
# FILTER COLLECTION TYPE
# -----------------------------
VALID_TYPES = ["Number/Light trap", "Percentage"]
df = df[df["Collection Type"].isin(VALID_TYPES)]


# -----------------------------
# GLOBAL SORT
# -----------------------------
df = df.sort_values([
    "PEST NAME",
    "Location",
    "Observation Year",
    "Standard Week"
])


# -----------------------------
# BUILD 5-LAG DATASET
# -----------------------------
rows = []

group_cols = ["PEST NAME", "Location"]

for (pest, loc), g in df.groupby(group_cols):

    g = g.reset_index(drop=True)

    pest_vals = g["Pest Value"].values

    # Need at least 6 weeks
    if len(g) < 6:
        continue

    for i in range(5, len(g) - 1):

        p_now  = max(pest_vals[i], 0)
        p_next = max(pest_vals[i+1], 0)

        p_lag1 = max(pest_vals[i-1], 0)
        p_lag2 = max(pest_vals[i-2], 0)
        p_lag3 = max(pest_vals[i-3], 0)
        p_lag4 = max(pest_vals[i-4], 0)
        p_lag5 = max(pest_vals[i-5], 0)

        # Skip inactive window
        if (
            p_now  < MIN_ACTIVE and
            p_next < MIN_ACTIVE and
            p_lag1 < MIN_ACTIVE and
            p_lag2 < MIN_ACTIVE and
            p_lag3 < MIN_ACTIVE and
            p_lag4 < MIN_ACTIVE and
            p_lag5 < MIN_ACTIVE
        ):
            continue

        row = {
            "pest": pest,
            "location": loc,
            "year": g.loc[i, "Observation Year"],
            "week": g.loc[i, "Standard Week"],

            # Lags
            "pest_t": p_now,
            "pest_lag1": p_lag1,
            "pest_lag2": p_lag2,
            "pest_lag3": p_lag3,
            "pest_lag4": p_lag4,
            "pest_lag5": p_lag5,

            # Climate
            "MaxT": g.loc[i, "MaxT"],
            "MinT": g.loc[i, "MinT"],
            "RH1": g.loc[i, "RH1(%)"],
            "RH2": g.loc[i, "RH2(%)"],
            "RF": g.loc[i, "RF(mm)"],
            "WS": g.loc[i, "WS(kmph)"],
            "SSH": g.loc[i, "SSH(hrs)"],
            "EVP": g.loc[i, "EVP(mm)"],

            # Target
            "pest_next": p_next,
            "delta_pest": p_next - p_now
        }

        rows.append(row)


# -----------------------------
# FINALIZE
# -----------------------------
final_df = pd.DataFrame(rows)

PEST_COLS = [
    "pest_t",
    "pest_lag1",
    "pest_lag2",
    "pest_lag3",
    "pest_lag4",
    "pest_lag5",
    "pest_next"
]

NO_LOG_PESTS = ["LeafBlast", "NeckBlast"]


def conditional_log(pest, value):

    if pest in NO_LOG_PESTS:
        return value

    return np.log1p(value)


for col in PEST_COLS:

    final_df[col] = final_df.apply(
        lambda r: conditional_log(r["pest"], r[col]),
        axis=1
    )


# -----------------------------
# SAVE
# -----------------------------
final_df.to_csv(OUTPUT_FILE, index=False)


# -----------------------------
# DOWNLOAD RESULT
# -----------------------------
files.download(OUTPUT_FILE)


# -----------------------------
# REPORT
# -----------------------------
print("Fixed 5-lag dataset created")
print("Rows:", len(final_df))
print("Saved to:", OUTPUT_FILE)

print("\nCheck pest scales:")
print(final_df[PEST_COLS].describe())

