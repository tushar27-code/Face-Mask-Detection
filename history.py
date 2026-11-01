
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "detection_history.csv"

HEADERS = [
    "Date",
    "Time",
    "Detection Mode",
    "Mask Count",
    "No Mask Count"
]


def save_detection(mode, mask_count, no_mask_count):

    file_exists = HISTORY_FILE.exists()

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(HEADERS)

        now = datetime.now()

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            mode,
            mask_count,
            no_mask_count
        ])


def read_history():

    if not HISTORY_FILE.exists():
        return []

    import pandas as pd

    return pd.read_csv(HISTORY_FILE)
