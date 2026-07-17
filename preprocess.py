import argparse
import json

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

from config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    CLASS_TO_INDEX,
    AAMI_CLASSES,
    DS1_TRAIN_RECORDS,
    DS1_VAL_RECORDS,
    DS2_RECORDS,
    PREFERRED_LEAD,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SAMPLING_RATE,
    SYMBOL_TO_AAMI,
    WINDOW_AFTER,
    WINDOW_BEFORE,
)


def _bandpass(signal):
    nyquist = 0.5 * SAMPLING_RATE
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist], btype="band")
    return filtfilt(b, a, signal)


def _select_lead(record):
    if PREFERRED_LEAD in record.sig_name:
        return record.p_signal[:, record.sig_name.index(PREFERRED_LEAD)]
    return record.p_signal[:, 0]


def _rr_features(samples):
    rr = np.diff(samples) / SAMPLING_RATE
    pre = np.concatenate([[np.nan], rr])
    post = np.concatenate([rr, [np.nan]])
    avg = np.nanmean(rr) if rr.size else 1.0
    kernel = np.ones(11) / 11.0
    padded = np.pad(pre, (5, 5), mode="edge")
    local = np.convolve(np.nan_to_num(padded, nan=avg), kernel, mode="valid")
    local = np.where(local > 0, local, avg)
    pre = np.nan_to_num(pre, nan=avg)
    post = np.nan_to_num(post, nan=avg)
    return np.stack([pre / avg, post / avg, pre / local, post / local], axis=1)


def segment_record(record_id):
    record = wfdb.rdrecord(str(RAW_DIR / record_id))
    annotation = wfdb.rdann(str(RAW_DIR / record_id), "atr")

    signal = _bandpass(_select_lead(record).astype(np.float64))
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    beats = [(s, CLASS_TO_INDEX[a]) for s, sym in zip(annotation.sample, annotation.symbol)
             if (a := SYMBOL_TO_AAMI.get(sym)) is not None]
    samples = np.array([b[0] for b in beats])
    rr = _rr_features(samples)

    windows, rr_kept, labels = [], [], []
    for i, (sample, label) in enumerate(beats):
        start, end = sample - WINDOW_BEFORE, sample + WINDOW_AFTER
        if start < 0 or end > signal.shape[0]:
            continue
        windows.append(signal[start:end])
        rr_kept.append(rr[i])
        labels.append(label)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(rr_kept, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def build_split(record_ids):
    per_record = {r: segment_record(r) for r in record_ids}
    x = np.concatenate([v[0] for v in per_record.values()], axis=0)
    x_rr = np.concatenate([v[1] for v in per_record.values()], axis=0)
    y = np.concatenate([v[2] for v in per_record.values()], axis=0)
    counts = {r: np.bincount(v[2], minlength=len(AAMI_CLASSES)).tolist() for r, v in per_record.items()}
    return x, x_rr, y, counts


def class_distribution(y):
    counts = np.bincount(y, minlength=len(AAMI_CLASSES))
    return {AAMI_CLASSES[i]: int(counts[i]) for i in range(len(AAMI_CLASSES))}


def main():
    parser = argparse.ArgumentParser(description="Segment MIT-BIH beats into a patient-independent DS1/DS2 dataset.")
    parser.add_argument("--output", default=str(PROCESSED_DIR / "mitbih_ds1ds2.npz"))
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, rr_train, y_train, train_counts = build_split(DS1_TRAIN_RECORDS)
    x_val, rr_val, y_val, val_counts = build_split(DS1_VAL_RECORDS)
    x_test, rr_test, y_test, test_counts = build_split(DS2_RECORDS)

    np.savez_compressed(
        args.output,
        x_train=x_train[..., np.newaxis], rr_train=rr_train, y_train=y_train,
        x_val=x_val[..., np.newaxis], rr_val=rr_val, y_val=y_val,
        x_test=x_test[..., np.newaxis], rr_test=rr_test, y_test=y_test,
    )

    summary = {
        "window_length": x_train.shape[1],
        "sampling_rate": SAMPLING_RATE,
        "splits": {
            "train": {"records": DS1_TRAIN_RECORDS, "n_beats": int(y_train.shape[0]), "class_distribution": class_distribution(y_train)},
            "val": {"records": DS1_VAL_RECORDS, "n_beats": int(y_val.shape[0]), "class_distribution": class_distribution(y_val)},
            "test": {"records": DS2_RECORDS, "n_beats": int(y_test.shape[0]), "class_distribution": class_distribution(y_test)},
        },
        "per_record_counts": {"train": train_counts, "val": val_counts, "test": test_counts},
    }
    summary_path = RESULTS_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved {args.output}")
    for split, info in summary["splits"].items():
        print(f"{split:5s}: {info['n_beats']:6d} beats  {info['class_distribution']}")
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
