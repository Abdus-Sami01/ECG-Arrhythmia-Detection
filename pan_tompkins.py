"""
Pan-Tompkins QRS detector (Pan & Tompkins 1985), streaming-compatible.

Implements the signal-processing chain:
  bandpass → derivative → squaring → moving-window integration → adaptive thresholding

The `PanTompkinsDetector` class processes one sample at a time, which maps
directly to a microcontroller interrupt-driven ISR.  `detect_beats` is the
offline batch wrapper used for evaluation.
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

from config import SAMPLING_RATE


# --- Offline batch detector (uses zero-phase filters for accuracy) ----------

def _bandpass_batch(signal, fs=SAMPLING_RATE):
    nyq = 0.5 * fs
    b, a = butter(1, [5 / nyq, 15 / nyq], btype="band")
    return filtfilt(b, a, signal)


def _derivative_batch(signal):
    # 5-point derivative per Pan & Tompkins
    kernel = np.array([-1, -2, 0, 2, 1]) * (SAMPLING_RATE / 8.0)
    return np.convolve(signal, kernel, mode="same")


def _moving_window_integrate(signal, fs=SAMPLING_RATE):
    # 150 ms window
    width = max(1, int(round(0.150 * fs)))
    kernel = np.ones(width) / width
    return np.convolve(signal, kernel, mode="same")


def _find_peaks_with_refractory(integrated, threshold, fs=SAMPLING_RATE):
    """Return sample indices of QRS peaks above threshold, enforcing 200 ms refractory."""
    refractory = int(0.200 * fs)
    peaks = []
    i = 1
    n = len(integrated)
    while i < n - 1:
        if integrated[i] > threshold and integrated[i] >= integrated[i - 1] and integrated[i] >= integrated[i + 1]:
            peaks.append(i)
            i += refractory
        else:
            i += 1
    return np.array(peaks, dtype=np.int64)


def detect_beats(signal, fs=SAMPLING_RATE):
    """
    Offline Pan-Tompkins detector.  Returns sample indices of detected R-peaks.

    Parameters
    ----------
    signal : 1-D float array, raw (unfiltered) ECG in mV or z-score units
    fs     : sampling rate (default SAMPLING_RATE from config)

    Returns
    -------
    peaks  : int64 array of R-peak sample positions
    """
    bp = _bandpass_batch(signal, fs)
    deriv = _derivative_batch(bp)
    squared = deriv ** 2
    integrated = _moving_window_integrate(squared, fs)

    # Initialise thresholds from the first 2 s
    init_samples = int(2.0 * fs)
    spki = np.max(integrated[:init_samples]) * 0.25  # signal peak estimate
    npki = np.mean(integrated[:init_samples]) * 0.5  # noise peak estimate
    threshold1 = npki + 0.25 * (spki - npki)

    peaks = _find_peaks_with_refractory(integrated, threshold1, fs)
    return peaks


# --- Streaming (sample-by-sample) detector ----------------------------------

class PanTompkinsDetector:
    """
    One-sample-at-a-time Pan-Tompkins detector suitable for microcontroller porting.

    Usage
    -----
    det = PanTompkinsDetector()
    for sample_index, raw_sample in enumerate(ecg_stream):
        r_peak = det.update(raw_sample, sample_index)
        if r_peak is not None:
            classify(r_peak)
    """

    # 150 ms integration window at 360 Hz → 54 samples
    WINDOW = int(round(0.150 * SAMPLING_RATE))
    REFRACTORY = int(0.200 * SAMPLING_RATE)

    def __init__(self):
        # Butterworth 1st-order bandpass 5–15 Hz causal (IIR)
        nyq = 0.5 * SAMPLING_RATE
        self._b, self._a = butter(1, [5 / nyq, 15 / nyq], btype="band")
        # IIR filter states
        self._zi_bp = np.zeros(max(len(self._b), len(self._a)) - 1)
        # Derivative delay line (5-point kernel needs 4 past samples)
        self._deriv_buf = np.zeros(5)
        # Integration circular buffer
        self._integ_buf = np.zeros(self.WINDOW)
        self._integ_idx = 0
        self._integ_sum = 0.0
        # Adaptive threshold state
        self._spki = 0.0
        self._npki = 0.0
        self._threshold1 = 0.0
        self._last_peak_sample = -self.REFRACTORY
        self._sample_count = 0
        self._initialized = False

    def update(self, raw_sample, sample_index):
        """
        Process one sample.  Returns detected R-peak sample index or None.
        """
        # 1. Bandpass (causal IIR)
        filtered, self._zi_bp = lfilter(self._b, self._a, [raw_sample], zi=self._zi_bp)
        bp_val = float(filtered[0])

        # 2. 5-point derivative
        self._deriv_buf = np.roll(self._deriv_buf, -1)
        self._deriv_buf[-1] = bp_val
        deriv_val = (SAMPLING_RATE / 8.0) * (
            -self._deriv_buf[0] - 2 * self._deriv_buf[1]
            + 2 * self._deriv_buf[3] + self._deriv_buf[4]
        )

        # 3. Squaring
        sq_val = deriv_val ** 2

        # 4. Moving-window integration (circular buffer)
        old = self._integ_buf[self._integ_idx]
        self._integ_sum += sq_val - old
        self._integ_buf[self._integ_idx] = sq_val
        self._integ_idx = (self._integ_idx + 1) % self.WINDOW
        integ_val = self._integ_sum / self.WINDOW

        self._sample_count += 1

        # Initialise thresholds from first 2 s
        if self._sample_count <= int(2.0 * SAMPLING_RATE):
            if integ_val > self._spki:
                self._spki = integ_val
            self._npki = max(self._npki, integ_val * 0.1)
            self._threshold1 = self._npki + 0.25 * (self._spki - self._npki)
            if self._sample_count == int(2.0 * SAMPLING_RATE):
                self._initialized = True
            return None

        if not self._initialized:
            return None

        # 5. Peak detection with refractory
        # (we can only confirm a peak on the next sample; simplified: detect above threshold)
        if (integ_val > self._threshold1
                and sample_index - self._last_peak_sample >= self.REFRACTORY):
            # Update signal/noise peak estimates
            self._spki = 0.125 * integ_val + 0.875 * self._spki
            self._threshold1 = self._npki + 0.25 * (self._spki - self._npki)
            self._last_peak_sample = sample_index
            return sample_index
        else:
            self._npki = 0.125 * integ_val + 0.875 * self._npki
            self._threshold1 = self._npki + 0.25 * (self._spki - self._npki)
            return None


# --- Evaluation against ground-truth annotations ----------------------------

def evaluate_detector(record_ids, tolerance_ms=150):
    """
    Compare detected R-peaks to ground-truth WFDB annotations.

    Returns dict with sensitivity, PPV, and F1 aggregated across all records.
    Tolerance window: ±tolerance_ms milliseconds.
    """
    import wfdb
    from config import RAW_DIR, PREFERRED_LEAD, SYMBOL_TO_AAMI

    tol = int(round(tolerance_ms / 1000 * SAMPLING_RATE))
    tp_total = fn_total = fp_total = 0

    for rec_id in record_ids:
        record = wfdb.rdrecord(str(RAW_DIR / rec_id))
        annotation = wfdb.rdann(str(RAW_DIR / rec_id), "atr")

        if PREFERRED_LEAD in record.sig_name:
            sig = record.p_signal[:, record.sig_name.index(PREFERRED_LEAD)]
        else:
            sig = record.p_signal[:, 0]
        sig = sig.astype(np.float64)

        # Ground-truth: any beat-type annotation mapped by SYMBOL_TO_AAMI
        gt = np.array([s for s, sym in zip(annotation.sample, annotation.symbol)
                       if sym in SYMBOL_TO_AAMI], dtype=np.int64)

        detected = detect_beats(sig)

        matched_gt = np.zeros(len(gt), dtype=bool)
        matched_det = np.zeros(len(detected), dtype=bool)

        for di, dp in enumerate(detected):
            diffs = np.abs(gt - dp)
            if diffs.size == 0:
                continue
            closest = np.argmin(diffs)
            if diffs[closest] <= tol and not matched_gt[closest]:
                matched_gt[closest] = True
                matched_det[di] = True

        tp = int(matched_det.sum())
        fp = int((~matched_det).sum())
        fn = int((~matched_gt).sum())
        tp_total += tp
        fp_total += fp
        fn_total += fn

    sens = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    ppv = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    return {
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "sensitivity": round(sens, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "tolerance_ms": tolerance_ms,
        "note": "simulated offline evaluation; streaming detector adds ~2 s initialisation latency",
    }


if __name__ == "__main__":
    import json
    from config import DS2_RECORDS, RESULTS_DIR

    print("Evaluating Pan-Tompkins detector on DS2 records...")
    result = evaluate_detector(DS2_RECORDS)
    print(f"  Sensitivity: {result['sensitivity']:.4f}")
    print(f"  PPV:         {result['ppv']:.4f}")
    print(f"  F1:          {result['f1']:.4f}")
    print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pan_tompkins_eval.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {RESULTS_DIR/'pan_tompkins_eval.json'}")
