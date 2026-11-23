# MIDI & Audio Music Classification

Three music-understanding tasks tackled with hybrid **classical + deep learning** ensembles. Each task pairs a hand-engineered tree/gradient-boosting model with a Transformer (or pretrained foundation model) and combines them at the logit/probability level. The result on every task is materially better than either component alone.

| Task | Problem | Input | Output | Final approach |
| --- | --- | --- | --- | --- |
| **1. Composer ID** | Identify the composer of a piano piece | MIDI file | 8-way single-label | 280-dim handcrafted features + `HistGradientBoosting`  **⊕**  5-layer Transformer over note-event tokens |
| **2. Next-Sequence Prediction** | Decide whether segment B plausibly follows segment A | Pair of MIDI segments | Binary | XGB + HGB + RandomForest stack on pair features  **⊕**  4-layer Transformer over `[CLS][SEPa] A [SEPb] B` |
| **3. Music Tagging** | Multi-label genre/mood tagging of music clips | 10 s MP3 | 10-way multi-label (`rock, oldies, jazz, pop, dance, blues, punk, chill, electronic, country`) | 3-seed ResNet34 on log-mel + Δ + ΔΔ  **⊕**  AudioSet-pretrained AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) fine-tune |

## Headline results (held-out test)

| Task | Metric | Score |
| --- | --- | --- |
| Task 1 (composer) | Accuracy | **0.663** |
| Task 3 (tagging)  | mAP      | **0.637** |

Task 2 was tuned with cross-validation; final logit-averaged ensemble is the artifact in `predictions2.json`.

---

## Repository layout

```
.
├── pipeline.py               # Main pipeline — Transformer / AST models + ensembling entry points
├── baseline.py               # Classical baseline pipeline (HGB / XGB stack / ResNet34)
├── baselines.ipynb           # Notebook version of the baseline pipeline
├── RUNBOOK.md                # Detailed run instructions and tuning knobs
├── run_unattended.ps1        # Unattended overnight training driver (Windows PowerShell)
├── writeup.txt               # One-paragraph-per-task technical summary
├── predictions1.json         # Task 1 final predictions
├── predictions2.json         # Task 2 final predictions
├── predictions3.json         # Task 3 final predictions
├── requirements.txt
└── README.md
```

The dataset (`student_files/`), all trained checkpoints (`*.pt`), and cached features (`cache_task1_audio/`) are intentionally **not** in the repo — see [Reproducing](#reproducing) below.

---

## Method details

### Task 1 — Composer classification (8 composers, 1 210 MIDIs)

Soft-vote ensemble of two models:

* **`Model1` (HGB).** Balanced `HistGradientBoostingClassifier` over a 280-dim per-piece feature vector:
  * chroma, duration-weighted chroma, velocity-weighted chroma
  * melodic-interval histogram and inter-onset-interval (IOI) histogram
  * velocity bins and log-duration bins
  * 12 × 12 pitch-class transition matrix
  * Krumhansl–Kessler key fits, polyphony, tempo, time signature

* **Transformer.** 5-layer pre-norm Transformer encoder (`d_model=192`, 6 heads, GELU) over per-note tokens. Each token embeds `pitch + log-binned duration + log-binned IOI + velocity bin + program-family`. A `[CLS]` token plus masked-mean pooling forms the classification head. Trained with class-balanced, label-smoothed cross-entropy and a chronological tail hold-out for model selection. 5-fold ensembled.

### Task 2 — Next-sequence prediction (binary)

Logit-averaged ensemble of two models, both evaluated with forward + reverse-order TTA:

* **Tree stack.** XGB + HGB + RandomForest on hand-engineered *pair* features:
  * per-segment head/tail/half-split summaries
  * tail-of-A → head-of-B "bridge" features (and the reverse)
  * full-segment pitch-class histograms, with their absolute difference and product
  * 4 metadata blocks: tempo, time signature, ticks-per-beat, program, total span

* **Pair Transformer.** 4-layer Transformer (`d_model=192`, 6 heads) over the concatenated sequence `[CLS] [SEPa] segA_notes [SEPb] segB_notes` with per-token segment embeddings. Augmented at train time with synthetic swaps; inference averages forward and reverse logits.

### Task 3 — Music tagging (10 tags, multi-label, 4 000 train clips)

Probability-averaged ensemble of:

* **ResNet34 tagger (3 seeds).** ImageNet-pretrained ResNet34 over 128-mel log-spectrograms with 1st / 2nd-order deltas (hop 512, 10 s clips). Trained with **asymmetric loss + positive-weighted BCE**, **mixup + cutmix**, **SpecAugment**, AdamW + cosine schedule with frozen-backbone warm-up, **EMA** model averaging, and **time-shift TTA** at inference.

* **AST fine-tune.** `MIT/ast-finetuned-audioset-10-10-0.4593` with a fresh 10-way multi-label head, BCE-with-logits, SpecAugment-style time/frequency masking, frozen-backbone warm-up, AdamW + cosine schedule with separate head / backbone learning rates.

The AST half is the headline: a single AST fine-tune in ~1 epoch on a 512-sample subset already reaches val mAP ≈ 0.57, comfortably ahead of the from-scratch ResNet ensemble (val ≈ 0.47, leaderboard ≈ 0.44). Ensembling the two gives the locked **0.637** test mAP.

---

## Reproducing

> **Hardware.** Developed on an 8 GB-VRAM consumer GPU (e.g. RTX 3070). The full overnight run takes ~3.5–4.5 hours end-to-end. A CPU-only run is possible for the baseline pipeline but not for the Transformer / AST halves.

### 1. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the Python 3.11 CPU environment used in CI:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

The optional `python scripts/to_json.py SOURCE DESTINATION` converter writes standard JSON to a new file.

### 2. Get the dataset

The course-provided dataset is **not** included. Drop it into `./student_files/` so the layout matches:

```
student_files/
├── task1_composer_classification/
│   ├── train.json     # {"midis/0.mid": composer_id, ...}
│   ├── test.json
│   └── midis/*.mid
├── task2_next_sequence_prediction/
│   ├── train.json
│   ├── test.json
│   └── midis/*.mid
└── task3_audio_classification/
    ├── train.json     # {"train/0.mp3": ["pop", ...], ...}
    ├── test.json
    ├── train/*.mp3
    └── test/*.mp3
```

### 3. Smoke test (~10–15 min)

```powershell
python -u -c "import pipeline as p; p.run_quick_v2()"
```

```bash
python -u -c "import pipeline as p; p.run_quick_v2()"
```

You should see Task 1 val tail acc > 0.45, Task 2 val acc > 0.55, and Task 3 AST val mAP > 0.55.

### 4. Full run (~3.5–4.5 h)

```powershell
python -u -c "import pipeline as p; p.run_all_v2()"
```

```bash
python -u -c "import pipeline as p; p.run_all_v2()"
```

This writes `predictions{1,2,3}.json`. See [`RUNBOOK.md`](RUNBOOK.md) for per-task entry points, ensemble-weight tuning, OOM workarounds, and how to re-ensemble without retraining.

### 5. Falling back to the classical baseline

```powershell
python -u -c "import baseline as a; a.run3_test_only_ensemble()"  # Task 3 from saved ResNet ckpts, no training
python -u -c "import baseline as a; a.run2()"                      # Task 2 tree stack
python -u -c "import baseline as a; a.run1()"                      # Task 1 HGB
```

```bash
python -u -c "import baseline as a; a.run3_test_only_ensemble()"  # Task 3 from saved ResNet ckpts, no training
python -u -c "import baseline as a; a.run2()"                      # Task 2 tree stack
python -u -c "import baseline as a; a.run1()"                      # Task 1 HGB
```

---

## Notes

* The `student_files/` dataset is **not** redistributed in this repo.
* Trained checkpoints (~2.2 GB total) are gitignored; they regenerate from `run_all_v2()`.
* Originally developed for a graduate music-information-retrieval course.
