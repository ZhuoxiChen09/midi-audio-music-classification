# Assignment 1 V2 — Run Instructions

`assignment1_v2.py` adds three neural extensions that ensemble with your existing solutions in `assignment1.py`:

| Task | What's new in V2 | Ensembled with |
|---|---|---|
| 1 | Small Transformer over MIDI note-event tokens (per-note pitch+dur+IOI+vel+program embeddings, chronological order) | Existing HGB (`Model1`) |
| 2 | Transformer over `[CLS][SEPa] segA_notes [SEPb] segB_notes` with segment embeddings | Existing XGB+HGB+RF stack (`Model2`) |
| 3 | **HuggingFace AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) multilabel fine-tune | Existing ResNet34 + EfficientNet checkpoints in cwd |

**Smoke test results** (already verified):

* Task 1: 2 epochs / 640 samples → val tail acc 0.43
* Task 2: 2 epochs / 512 samples → val acc 0.56
* Task 3 AST: **1 epoch / 512 samples → val mAP 0.57** (vs. your full ResNet34 ensemble val 0.47, leaderboard test 0.44)

The Task 3 result is the headline — AudioSet-pretrained AST is a different league than a from-scratch ResNet for music tagging.

---

## Step 1 — Light run (sanity check, ~10–15 min)

Confirms each model trains and improves over your current numbers:

```powershell
python -u -c "import assignment1 as v2; v2.run_quick_v2()"
```

This trains tiny subsets for each task, writes `predictions{1,2,3}.json` (subset only, not for submission), and prints val scores. If you see:
- Task 1 val tail acc > 0.45
- Task 2 val acc > 0.55
- Task 3 AST val mAP > 0.55

…then the full run is worth it.

## Step 2 — Full overnight run

```powershell
python -u -c "import assignment1 as v2; v2.run_all_v2()"
```

Defaults: T1 35 epochs, T2 18 epochs, T3 AST 12 epochs. Writes `predictions1.json`, `predictions2.json`, `predictions3.json` ready to submit.

**Expected wall-clock on RTX with 8 GB VRAM:**
- Task 1 (Transformer + HGB): ~20–30 min
- Task 2 (Transformer + tree stack): ~25–40 min
- Task 3 (AST + ResNet inference ensemble): ~2.5–3.5 h
- **Total: ~3.5–4.5 h**

---

## Per-task run (if you only want to update one)

```powershell
# Task 1 only — uses cached Transformer if available
python -u -c "import assignment1 as v2; v2.run1_v2()"

# Task 2 only
python -u -c "import assignment1 as v2; v2.run2_v2()"

# Task 3 only — re-uses your existing task3_checkpoint_*.pt for the ResNet half
python -u -c "import assignment1 as v2; v2.run3_v2()"
```

To re-ensemble without re-training (after the first full run):

```powershell
python -u -c "import assignment1 as v2; v2.run3_v2(skip_ast_train=True)"
```

---

## Tunable ensemble weights

The defaults are conservative. After you see val numbers, you may want to push weight toward whichever model wins on val:

```powershell
# Task 1: give Transformer more weight if its val tail acc beats HGB diagnostics
python -u -c "import assignment1 as v2; v2.run1_v2(hgb_weight=0.4, tfm_weight=0.6)"

# Task 3: AST dominates — push its weight up
python -u -c "import assignment1 as v2; v2.run3_v2(ast_weight=0.75, resnet_weight=0.25, skip_ast_train=True)"
```

## Environment knobs (for tight memory)

```powershell
$env:TASK3_AST_BATCH = "4"        # default 8; drop to 4 if OOM
$env:TASK3_AST_EVAL_BATCH = "4"
```

If you ever OOM, kill the run, set the env vars above, then re-run with `skip_*_train=True` to resume from saved checkpoints.

---

## What gets saved

```
task1_v2_transformer.pt   # Task 1 Transformer state + class list
task2_v2_transformer.pt   # Task 2 Pair Transformer state
task3_v2_ast.pt           # Task 3 AST fine-tuned state
predictions1.json         # final submission for Task 1
predictions2.json         # final submission for Task 2
predictions3.json         # final submission for Task 3
```

## Updating writeup.txt after the full run

Add (replace existing) the V2 paragraphs:

> **Task 1:** As before — HGB on 280-D MIDI histogram features — **plus** a 5-layer Transformer (d_model=192, 6 heads) over per-note (pitch, dur-bin, IOI-bin, vel-bin, program) embeddings with a CLS token, soft-voted with the HGB.
>
> **Task 2:** Tree ensemble (XGB+HGB+RF) on pair features — **plus** a 4-layer Transformer over `[CLS][SEPa]segA[SEPb]segB` with segment embeddings, trained with forward+reverse augmentation, logit-averaged with the tree predictions.
>
> **Task 3:** ResNet34 3-seed ensemble on log-mel+deltas — **plus** AudioSet-pretrained AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) fine-tuned for multilabel with frozen-head warmup, ASL-style SpecAugment, AdamW + cosine LR, ensembled (probability average) with the ResNet stack.

---

## Falling back

The original baseline pipeline is preserved in `assignment1_baseline.py`. If V2 breaks:

```powershell
# Regenerate Task 3 predictions from saved ResNet/EffNet checkpoints (no training)
python -u -c "import assignment1_baseline as a; a.run3_test_only_ensemble()"

# Regenerate Task 2 from the tree stack (retrains a few min)
python -u -c "import assignment1_baseline as a; a.run2()"

# Regenerate Task 1 from the HGB stack
python -u -c "import assignment1_baseline as a; a.run1()"
```
