import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from config import AAMI_CLASSES


def macro_f1(y_true, y_pred, labels=None):
    labels = labels if labels is not None else list(range(len(AAMI_CLASSES)))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(AAMI_CLASSES))))
    total = cm.sum()
    report = {}
    for i, name in enumerate(AAMI_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        support = int(cm[i, :].sum())
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0.0
        report[name] = {
            "support": support,
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "ppv": round(float(ppv), 4),
            "f1": round(float(f1), 4),
        }

    present = [i for i in range(len(AAMI_CLASSES)) if cm[i, :].sum() > 0]
    nsvf = [i for i in range(len(AAMI_CLASSES)) if AAMI_CLASSES[i] in ("N", "S", "V", "F")]
    return {
        "per_class": report,
        "macro_f1_5class": macro_f1(y_true, y_pred, labels=present),
        "macro_f1_nsvf": macro_f1(y_true, y_pred, labels=nsvf),
        "accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "confusion_matrix": cm.tolist(),
    }
