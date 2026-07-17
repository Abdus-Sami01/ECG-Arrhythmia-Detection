# ECG Arrhythmia Detection — GRU for Microcontroller Deployment

On-device beat-level arrhythmia classification sized for a wearable-class microcontroller
(ARM Cortex-M4, ~256 KB–1 MB RAM, no FPU reliance, coin-cell power budget). The goal is not
peak leaderboard accuracy — it is a defensible answer to one sentence:

> **X % macro-F1 at Y KB model size at Z ms inference on a Cortex-M4-class chip.**

## Why on-device
A wearable arrhythmia monitor cannot depend on a cloud round-trip: latency, patient-data
privacy, and battery all argue for classifying each beat on the device itself. That constraint —
not the model architecture — is the subject of this project.

## What makes this rigorous
- **Patient-independent (inter-patient) split.** Train and test never share a patient
  (de Chazal et al. 2004 DS1/DS2). Random beat-level splits leak patient identity and inflate
  accuracy; this project refuses that shortcut. See [`data/DATA_CARD.md`](data/DATA_CARD.md).
- **AAMI 5-class scheme** (N, S, V, F, Q), not raw MIT-BIH symbols.
- **GRU over LSTM** for a smaller parameter/state footprint under the RAM budget — a claim
  tested directly against an LSTM baseline (Phase 4).
- **Measured, not asserted, compression.** Every shrink/quantization step reports the
  before/after macro-F1 cost on the same held-out patients.

## Status
- [x] **Phase 1 — Data & preprocessing:** authoritative WFDB download, 0.5–40 Hz bandpass,
  per-record z-score, 180-sample beat windows, AAMI mapping, patient-independent DS1/DS2 split.
- [ ] Phase 2 — Full-precision GRU training with class-weighted loss and macro-F1 model selection.
- [ ] Phase 3 — Architecture shrink + int8 quantization + edge memory/latency estimation.
- [ ] Phase 4 — CNN / LSTM / fully-connected baselines at matched size.

## Reproduce Phase 1
```bash
pip install -r requirements.txt
python download_data.py      # WFDB files -> data/raw/mitdb/
python preprocess.py         # -> data/processed/mitbih_ds1ds2.npz + results/dataset_summary.json
```

## Repository layout
```
config.py           single source of truth: split, AAMI mapping, window, paths
download_data.py    MIT-BIH WFDB acquisition (idempotent)
preprocess.py       filter, segment, patient-independent split -> .npz
data/DATA_CARD.md   split protocol, label mapping, class distribution, caveats
```

## Honest limitations (carried forward, expanded per phase)
- Single-lead (MLII) only — not a multi-lead clinical montage.
- R-peak locations are ground-truth annotations, not a real-time detector (a documented
  stretch goal), so this is a classification study, not an end-to-end streaming claim.
- The Q class is near-empty under the non-paced inter-patient protocol; its per-class metrics
  are reported with that caveat.
- Not clinically validated.
