from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "mitdb"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

MITDB_MIRROR = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0"

SAMPLING_RATE = 360
WINDOW_BEFORE = 90
WINDOW_AFTER = 90
WINDOW_LENGTH = WINDOW_BEFORE + WINDOW_AFTER
RR_FEATURES = 4
GRU_TIMESTEPS = 18

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 3

PREFERRED_LEAD = "MLII"

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(AAMI_CLASSES)}

SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

PACED_RECORDS = ["102", "104", "107", "217"]

DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

DS1_VAL_RECORDS = ["118", "124", "205", "223"]
DS1_TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in DS1_VAL_RECORDS]

ALL_RECORDS = sorted(DS1_RECORDS + DS2_RECORDS + PACED_RECORDS, key=int)
