import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, RESULTS_DIR


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gru = json.loads((RESULTS_DIR / "headline_table.json").read_text())
    baselines = json.loads((RESULTS_DIR / "baseline_comparison.json").read_text())

    cnn = json.loads((RESULTS_DIR / "cnn_sweep.json").read_text())
    points = [(r["model"], r["int8_size_kb"], r["macro_f1_int8"], "GRU sweep") for r in gru]
    points += [(r["model"], r["int8_size_kb"], r["macro_f1_int8"], "CNN sweep") for r in cnn]
    for r in baselines:
        if r["model"] in ("gru16", "cnn16"):
            continue
        family = "LSTM" if r["model"].startswith("lstm") else "FC"
        points.append((r["model"], r["int8_size_kb"], r["macro_f1_int8"], family))

    colors = {"GRU sweep": "#1f77b4", "CNN sweep": "#2ca02c", "LSTM": "#ff7f0e", "FC": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for family in colors:
        pts = [p for p in points if p[3] == family]
        if pts:
            ax.scatter([p[1] for p in pts], [p[2] for p in pts], c=colors[family], label=family, s=70, zorder=3)
    for name, size, f1, _ in points:
        ax.annotate(name, (size, f1), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("int8 model size (KB, log scale)")
    ax.set_ylabel("DS2 macro-F1 (N,S,V,F)")
    ax.set_title("Accuracy vs. int8 size on held-out patients (DS2)")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_size.png", dpi=150)
    print(f"wrote {FIGURES_DIR/'accuracy_vs_size.png'}")


if __name__ == "__main__":
    main()
