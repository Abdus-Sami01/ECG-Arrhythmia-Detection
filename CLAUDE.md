# CLAUDE.md — ECG edge-arrhythmia project law

## Absolute rules (non-negotiable)
1. **Nothing stays untracked.** After every unit of work `git status` is clean: code and docs committed, generated artifacts (`data/`, `models/`, `results/`, `figures/`) covered by `.gitignore`. No stray files.
2. **Commits are one-liner messages, authored as the owner, never co-authored, and land verified.** No `Co-Authored-By` trailer, no "Claude" attribution, and the word "Claude" never appears in any git operation — message, branch name, tag, or otherwise. Author identity is `abdus-sami01 <muhammadabdulsami7@gmail.com>`. The owner's GPG private key (`315ECD1899DE017F`) is not present in the build sandbox and must never be pasted in, so commits are created through the GitHub API, which signs them with GitHub's key and shows them **Verified**. Branch is `main` and generic names only — never a `claude/`-prefixed branch.
3. **Comments only when the code genuinely cannot speak for itself, and then one line.** No narration of what the next line does. A comment carries a constraint, a citation, or a non-obvious reason — nothing else.
4. **Excellence over speed.** Iteration count and elapsed time are free; correctness and rigor are the only targets. Never rush a result into place; get it right.

## Methodology guardrails (the rigor that makes this project credible)
- **The patient-independent split is sacred.** DS1/DS2 record membership (`config.py`) is never mixed; no record spans two splits; beats are never split randomly across the whole dataset. This is the project's central methodological claim — protect it.
- **Never report bare accuracy.** The data is ~86–91% class N. Report per-class sensitivity, specificity, PPV, and macro-F1. Confusion matrices on DS2.
- **Handle imbalance with class-weighted loss**, not synthetic oversampling of waveforms.
- **Quantization is measured, not asserted.** Every compression step reports before/after macro-F1 on the same DS2 test set; the accuracy cost is stated numerically.
- **Simulated vs. measured is stated explicitly.** Any latency/memory number that is estimated rather than measured on hardware is labeled as such.
- **One source of truth.** The split, the AAMI mapping, window size, and paths live only in `config.py`. Never hardcode them elsewhere.

## Working loop
- Verify end-to-end before claiming done: run the code, show the numbers.
- State the actual outcome, including failures and skipped steps, plainly.
- A change to preprocessing invalidates downstream artifacts — re-run the affected stages, do not leave stale `.npz`/models.
