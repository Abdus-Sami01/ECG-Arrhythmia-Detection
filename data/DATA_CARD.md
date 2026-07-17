# Data Card — MIT-BIH Arrhythmia Database (edge-deployment preprocessing)

## Source
- **Dataset:** MIT-BIH Arrhythmia Database, version 1.0.0 (48 half-hour two-lead ambulatory ECG recordings, 360 Hz).
- **Acquisition:** PhysioNet Open Data mirror on AWS S3 (`https://physionet-open.s3.amazonaws.com/mitdb/1.0.0`), a byte-identical mirror of the PhysioNet `mitdb` release. Retrieved by `download_data.py`; the PhysioNet host itself was unreachable from the build environment, the S3 mirror is the same authoritative data.
- **License:** Open Data Commons Attribution (ODC-By) / PhysioNet open access.

## Signal selection
- Single lead per record, **MLII preferred** (present in all 44 non-paced records), falling back to channel 0 when MLII is absent. Record 114 carries MLII in channel 1 and is handled by name lookup, not index.

## Preprocessing pipeline (`preprocess.py`)
1. **Bandpass filter:** Butterworth, order 3, 0.5–40 Hz, zero-phase (`scipy.signal.filtfilt`). Removes baseline wander (<0.5 Hz) and high-frequency noise (>40 Hz).
2. **Per-record z-score normalization:** each record's filtered lead is standardized by its own mean and standard deviation. This matches on-device reality — a wearable never observes global dataset statistics — and controls for inter-patient amplitude variation.
3. **Beat segmentation:** for every annotated beat, a fixed 180-sample window (90 before, 90 after the annotated R-peak ≈ 0.5 s) is extracted. R-peak locations come from the database's ground-truth annotations, not a detector (detector robustness is a documented stretch goal). Beats whose window would cross a record boundary are dropped.
4. **RR-interval features (auxiliary input):** four dimensionless timing features per beat — pre-RR and post-RR each divided by the record's average RR and by a local 11-beat average RR. Supraventricular (S) beats are defined largely by *prematurity*, not morphology, and a single centered beat window is blind to timing; these features are the standard remedy (de Chazal et al. 2004) and are trivially computable on-device from R-peak spacing with a running RR average. Measured class separation on the training set: pre-RR/avg ≈ 1.03 (N), 0.70 (S), 0.71 (V), 0.95 (F).
5. **Label mapping:** MIT-BIH annotation symbols are collapsed to the AAMI 5 superclasses. Non-beat annotations (rhythm, signal-quality, and noise markers such as `+ ~ | x [ ] ! "`) are not beats and are skipped.

### AAMI label mapping
| AAMI class | MIT-BIH symbols |
|---|---|
| N — Normal | N, L, R, e, j |
| S — Supraventricular ectopic | A, a, J, S |
| V — Ventricular ectopic | V, E |
| F — Fusion | F |
| Q — Unknown/paced | /, f, Q |

## Patient-independent split (inter-patient, de Chazal et al. 2004)
Records are partitioned by **patient**, never by beat — no record appears in more than one split. The 4 paced records (102, 104, 107, 217) are excluded per AAMI recommendation. DS1 is used for training and validation, DS2 is the held-out test set.

- **Train (DS1 minus val):** 101, 106, 108, 109, 112, 114, 115, 116, 119, 122, 201, 203, 207, 208, 209, 215, 220, 230
- **Validation (DS1 subset):** 118, 124, 205, 223
- **Test (DS2):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

The train/validation division within DS1 is itself patient-independent so that validation macro-F1 (the early-stopping and model-selection signal) estimates generalization to unseen patients rather than unseen beats. The validation records were chosen to carry meaningful S and F support (203 S, 30 F): those minority classes are concentrated in a few patients — 372 of the ~415 DS1 fusion beats belong to a single record (208), which is kept in training — so a naive validation split yields a near-empty, noise-dominated minority signal. The chosen split keeps 208 and the S-rich records in training while still giving validation enough minority beats to make macro-F1 model selection stable.

## Class distribution (beats)
| Split | N | S | V | F | Q | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 37539 | 741 | 3181 | 384 | 8 | 41853 |
| Val | 8317 | 203 | 607 | 30 | 0 | 9157 |
| Test | 44246 | 1837 | 3220 | 388 | 7 | 49698 |

## Known caveats
- **Severe imbalance:** N is ~86–91% of every split. Overall accuracy is therefore not a meaningful metric; evaluation uses per-class sensitivity/specificity and macro-F1.
- **Q is near-empty in this protocol.** Most Q beats are paced (`/`) beats, which live almost entirely in the 4 excluded paced records. Under the inter-patient, non-paced protocol the Q class has single-digit support (8 / 0 / 7). It is retained to keep the AAMI 5-class scheme intact, but per-class Q metrics are not statistically meaningful and are reported with that caveat.
- **S and F are the hard minority classes** and are where quantization is most likely to degrade — watched explicitly in Phase 3.
- R-peak locations are ground-truth annotations, not detector output; this is a deliberate scope boundary for the classification study, not a real-time detection claim.
