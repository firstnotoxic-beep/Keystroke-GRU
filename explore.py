"""Exploratory data analysis on the CMU DSL keystroke dataset (EDA only)."""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from config import DSL_DATA_PATH

# NOTE: explore.py reads the raw DSL CSV schema (subject, H.period, DD.*, etc.)
# which differs from the processed pipeline schema (H1..UD10, Label).
# This script is for EDA only and is not part of the ML pipeline.


def main() -> None:
    if not DSL_DATA_PATH.is_file():
        print(f"ERROR: Dataset file not found: {DSL_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DSL_DATA_PATH)

    print(df.head())
    df.info()

    plt.hist(df["H.period"], bins=50, edgecolor="black")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
