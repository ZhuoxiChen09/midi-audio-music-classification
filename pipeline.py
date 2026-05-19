"""Main pipeline — deep-learning extensions over the classical baseline.

Three neural models ensembled with the classical pipeline in `baseline.py`:

  * Task 1: small Transformer encoder over MIDI note-event tokens
            (per-note pitch + duration-bin + IOI-bin + velocity-bin embeddings,
            chronological order). Soft-vote ensemble with the existing HGB.
  * Task 2: Transformer over [seg_a notes] + [SEP] + [seg_b notes] for
            symmetric pair classification. Logit-averaged ensemble with the
            existing XGB/HGB/RF stack and the forward+reverse trick.
  * Task 3: HuggingFace AST (MIT/ast-finetuned-audioset-10-10-0.4593)
            fine-tuned for multilabel music tagging. Probability-averaged
            ensemble with the existing ResNet34 / EfficientNet checkpoints.

All entry points support a "quick" mode that completes in a few minutes on
8 GB VRAM so you can sanity-check validation gains before launching an
overnight full run.

Entry points:
    run1_v2(quick=False)        # Task 1 train + ensemble + write predictions1.json
    run2_v2(quick=False)        # Task 2 train + ensemble + write predictions2.json
    run3_v2(quick=False)        # Task 3 train AST + ensemble with existing + write predictions3.json
    run_all_v2(quick=False)     # all three
    run_quick_v2()              # alias for run_all_v2(quick=True)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import miditoolkit

warnings.filterwarnings("ignore")

# Re-use everything from the baseline module so we don't fork it.
import baseline as a1
from baseline import (
    DATAROOT1,
    DATAROOT2,
    DATAROOT3,
    Model1,
    Model2,
    TAGS,
    write_submission_predictions,
    _task3_checkpoint_path,
    _load_task3_checkpoint,
    _make_test_loader,
    _Loaders,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


def _log_bin(value: float, lo: float, hi: float, bins: int) -> int:
    """Map a positive value to a log-spaced bin id in [0, bins-1]."""
    if value <= 0 or not math.isfinite(value):
        return 0
    v = math.log(max(value, 1e-4))
    width = (hi - lo) / bins
    if width <= 0:
        return 0
    idx = int((v - lo) / width)
    return max(0, min(bins - 1, idx))


# ===========================================================================
# Task 1 — MIDI note-event Transformer
# ===========================================================================
T1_PITCH_DIM = 128                  # 0..127 + we reserve 0 for pad
T1_DUR_BINS = 16
T1_IOI_BINS = 16
T1_VEL_BINS = 8
T1_PROG_BINS = 16
T1_PAD_ID = 0
T1_CLS_ID = 1                       # we add 2 special ids in the pitch space
T1_PITCH_OFFSET = 2                 # pitch 0 -> id 2

T1_MAX_LEN = 384                    # notes per piece (plus 1 CLS)
T1_DUR_LO, T1_DUR_HI = math.log(0.05), math.log(8.0)
T1_IOI_LO, T1_IOI_HI = math.log(0.02), math.log(4.0)

T1_D_MODEL = 192
T1_N_HEADS = 6
T1_N_LAYERS = 5
T1_FFN = 512
T1_DROPOUT = 0.2


def _tokenize_midi_task1(path: str) -> dict:
    """Return per-note token arrays for one MIDI file (no CLS prefix)."""
    midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT1, path))
    tpb = max(midi.ticks_per_beat, 1)

    notes = []
    programs = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        prog = int(getattr(inst, "program", 0)) // 8  # 16 family bins
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity, prog))
            programs.append(prog)
    if not notes:
        return None
    notes.sort(key=lambda t: (t[0], t[2]))

    n = len(notes)
    pitch_ids = np.zeros(n, dtype=np.int64)
    dur_ids = np.zeros(n, dtype=np.int64)
    ioi_ids = np.zeros(n, dtype=np.int64)
    vel_ids = np.zeros(n, dtype=np.int64)
    prog_ids = np.zeros(n, dtype=np.int64)

    last_start = None
    for i, (s, e, p, v, prog) in enumerate(notes):
        beats_dur = max((e - s) / tpb, 1e-3)
        beats_ioi = 0.0 if last_start is None else max((s - last_start) / tpb, 0.0)
        last_start = s
        pitch_ids[i] = int(p) + T1_PITCH_OFFSET
        dur_ids[i] = _log_bin(beats_dur, T1_DUR_LO, T1_DUR_HI, T1_DUR_BINS)
        ioi_ids[i] = _log_bin(max(beats_ioi, 1e-3), T1_IOI_LO, T1_IOI_HI, T1_IOI_BINS)
        vel_ids[i] = int(min(7, max(0, v // 16)))
        prog_ids[i] = int(min(T1_PROG_BINS - 1, max(0, prog)))

    return {
        "pitch": pitch_ids,
        "dur": dur_ids,
        "ioi": ioi_ids,
        "vel": vel_ids,
        "prog": prog_ids,
        "n": n,
    }


def _t1_collate(batch: list, max_len: int = T1_MAX_LEN):
    """Random crop / left-pad note sequences; prepend a CLS token."""
    B = len(batch)
    L = max_len + 1  # +1 for CLS
    pitch = np.zeros((B, L), dtype=np.int64)
    dur = np.zeros((B, L), dtype=np.int64)
    ioi = np.zeros((B, L), dtype=np.int64)
    vel = np.zeros((B, L), dtype=np.int64)
    prog = np.zeros((B, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    labels = np.zeros(B, dtype=np.int64)
    is_cls = np.zeros((B, L), dtype=bool)
    for i, (item, y) in enumerate(batch):
        if item is None:
            pitch[i, 0] = T1_CLS_ID
            mask[i, 0] = True
            is_cls[i, 0] = True
            labels[i] = y
            continue
        n = item["n"]
        if n > max_len:
            start = np.random.randint(0, n - max_len + 1) if y is not None else (n - max_len) // 2
            sl = slice(start, start + max_len)
            take = max_len
        else:
            sl = slice(0, n)
            take = n
        pitch[i, 0] = T1_CLS_ID
        is_cls[i, 0] = True
        pitch[i, 1 : 1 + take] = item["pitch"][sl]
        dur[i, 1 : 1 + take] = item["dur"][sl]
        ioi[i, 1 : 1 + take] = item["ioi"][sl]
        vel[i, 1 : 1 + take] = item["vel"][sl]
        prog[i, 1 : 1 + take] = item["prog"][sl]
        mask[i, : 1 + take] = True
        labels[i] = y
    return {
        "pitch": torch.from_numpy(pitch),
        "dur": torch.from_numpy(dur),
        "ioi": torch.from_numpy(ioi),
        "vel": torch.from_numpy(vel),
        "prog": torch.from_numpy(prog),
        "mask": torch.from_numpy(mask),
        "is_cls": torch.from_numpy(is_cls),
        "labels": torch.from_numpy(labels),
    }


class _Task1Dataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int] | None, cache: dict):
        self.paths = paths
        self.labels = labels
        self.cache = cache

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if p not in self.cache:
            self.cache[p] = _tokenize_midi_task1(p)
        y = -1 if self.labels is None else int(self.labels[idx])
        return self.cache[p], y


class _MidiTransformerT1(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        d = T1_D_MODEL
        self.pitch_emb = nn.Embedding(T1_PITCH_DIM + 2, d, padding_idx=T1_PAD_ID)
        self.dur_emb = nn.Embedding(T1_DUR_BINS, d)
        self.ioi_emb = nn.Embedding(T1_IOI_BINS, d)
        self.vel_emb = nn.Embedding(T1_VEL_BINS, d)
        self.prog_emb = nn.Embedding(T1_PROG_BINS, d)
        self.pos_emb = nn.Embedding(T1_MAX_LEN + 2, d)
        self.input_norm = nn.LayerNorm(d)
        self.input_drop = nn.Dropout(T1_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=T1_N_HEADS, dim_feedforward=T1_FFN,
            dropout=T1_DROPOUT, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=T1_N_LAYERS)
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Dropout(T1_DROPOUT),
            nn.Linear(d, n_classes),
        )

    def forward(self, pitch, dur, ioi, vel, prog, mask, is_cls):
        B, L = pitch.shape
        pos = torch.arange(L, device=pitch.device).unsqueeze(0).expand(B, L)
        x = (
            self.pitch_emb(pitch)
            + self.dur_emb(dur)
            + self.ioi_emb(ioi)
            + self.vel_emb(vel)
            + self.prog_emb(prog)
            + self.pos_emb(pos)
        )
        x = self.input_drop(self.input_norm(x))
        # PyTorch nn.TransformerEncoder: True = ignore.
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Pool: CLS position + masked mean.
        cls = x[:, 0, :]
        m = mask.float().unsqueeze(-1)
        m_sum = m.sum(dim=1).clamp(min=1.0)
        avg = (x * m).sum(dim=1) / m_sum
        return self.head(self.head_norm(cls + avg))


def _read_task1_train() -> Tuple[List[str], List[int]]:
    with open(os.path.join(DATAROOT1, "train.json")) as f:
        d = eval(f.read())
    paths = list(d.keys())
    labels = [int(d[p]) for p in paths]
    return paths, labels


def _read_task1_test() -> List[str]:
    with open(os.path.join(DATAROOT1, "test.json")) as f:
        d = eval(f.read())
    return list(d.keys()) if isinstance(d, dict) else list(d)


def train_task1_transformer(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.02,
    val_ratio: float = 0.12,
    quick: bool = False,
    save_path: str = "task1_v2_transformer.pt",
) -> dict:
    """Train the Task 1 Transformer. Returns dict with state, classes, val acc."""
    device = _device()
    paths, labels = _read_task1_train()
    classes = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_idx[c] for c in labels], dtype=np.int64)

    if quick:
        # Light subset for smoke test
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(paths))[: min(640, len(paths))]
        paths = [paths[i] for i in idx]
        y_all = y_all[idx]
        epochs = min(epochs, 5)
        batch_size = min(batch_size, 24)

    n = len(paths)
    val_n = max(8, int(n * val_ratio))
    # Use tail (chronological) holdout — matches existing diagnostics.
    tr_idx = np.arange(0, n - val_n)
    va_idx = np.arange(n - val_n, n)

    cache: dict = {}
    train_ds = _Task1Dataset([paths[i] for i in tr_idx], [y_all[i] for i in tr_idx], cache)
    val_ds = _Task1Dataset([paths[i] for i in va_idx], [y_all[i] for i in va_idx], cache)

    def _make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: _t1_collate(b, T1_MAX_LEN),
        )

    train_loader = _make_loader(train_ds, True)
    val_loader = _make_loader(val_ds, False)

    model = _MidiTransformerT1(len(classes)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Class-balanced loss
    cls_count = np.bincount(y_all[tr_idx], minlength=len(classes)).astype(np.float32)
    cls_w = (cls_count.mean() / np.maximum(cls_count, 1.0)).clip(0.25, 4.0)
    weight = torch.tensor(cls_w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    total_steps = epochs * max(1, len(train_loader))
    warmup = max(50, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())

    best_val = -1.0
    best_state = None
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"T1-V2 epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                logits = model(
                    batch["pitch"], batch["dur"], batch["ioi"],
                    batch["vel"], batch["prog"], batch["mask"], batch["is_cls"],
                )
                loss = loss_fn(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            step += 1

        # Val
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    logits = model(
                        batch_d["pitch"], batch_d["dur"], batch_d["ioi"],
                        batch_d["vel"], batch_d["prog"], batch_d["mask"],
                        batch_d["is_cls"],
                    )
                all_logits.append(logits.float().cpu())
                all_labels.append(batch_d["labels"].cpu())
        val_logits = torch.cat(all_logits)
        val_y = torch.cat(all_labels)
        val_pred = val_logits.argmax(dim=1)
        val_acc = float((val_pred == val_y).float().mean())
        print(f"[T1-V2] epoch {epoch+1} loss={running/max(1,len(train_loader)):.4f} val_tail_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"state": best_state or model.state_dict(),
         "classes": classes,
         "val_tail_acc": best_val},
        save_path,
    )
    print(f"[T1-V2] saved {save_path} | best tail acc={best_val:.4f}")
    return {"state": best_state or model.state_dict(), "classes": classes, "val_tail_acc": best_val}


def train_task1_transformer_kfold(
    k_folds: int = 3,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.02,
    quick: bool = False,
    save_prefix: str = "task1_v2_transformer_fold",
) -> List[dict]:
    """Train K-fold transformer ensemble. Returns list of ckpts."""
    device = _device()
    paths, labels = _read_task1_train()
    classes = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_idx[c] for c in labels], dtype=np.int64)

    if quick:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(paths))[: min(640, len(paths))]
        paths = [paths[i] for i in idx]
        y_all = y_all[idx]
        epochs = min(epochs, 5)
        batch_size = min(batch_size, 24)

    n = len(paths)
    rng = np.random.RandomState(SEED + 7)
    perm = rng.permutation(n)
    fold_size = n // k_folds
    ckpts = []
    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n
        va_idx = perm[val_start:val_end]
        tr_idx = np.concatenate([perm[:val_start], perm[val_end:]])

        cache: dict = {}
        train_ds = _Task1Dataset([paths[i] for i in tr_idx], [int(y_all[i]) for i in tr_idx], cache)
        val_ds = _Task1Dataset([paths[i] for i in va_idx], [int(y_all[i]) for i in va_idx], cache)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                                  collate_fn=lambda b: _t1_collate(b, T1_MAX_LEN))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                                collate_fn=lambda b: _t1_collate(b, T1_MAX_LEN))

        torch.manual_seed(SEED + 100 * fold)
        model = _MidiTransformerT1(len(classes)).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        cls_count = np.bincount(y_all[tr_idx], minlength=len(classes)).astype(np.float32)
        cls_w = (cls_count.mean() / np.maximum(cls_count, 1.0)).clip(0.25, 4.0)
        weight = torch.tensor(cls_w, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

        total_steps = epochs * max(1, len(train_loader))
        warmup = max(50, total_steps // 20)

        def lr_lambda(step, _warmup=warmup, _total=total_steps):
            if step < _warmup:
                return step / max(1, _warmup)
            prog = (step - _warmup) / max(1, _total - _warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())
        best_val, best_state = -1.0, None
        for epoch in range(epochs):
            model.train()
            running = 0.0
            for batch in tqdm(train_loader, desc=f"T1-V2 KF{fold+1}/{k_folds} ep{epoch+1}/{epochs}"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    logits = model(batch["pitch"], batch["dur"], batch["ioi"],
                                   batch["vel"], batch["prog"], batch["mask"], batch["is_cls"])
                    loss = loss_fn(logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                running += float(loss.detach())
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                        logits = model(batch_d["pitch"], batch_d["dur"], batch_d["ioi"],
                                       batch_d["vel"], batch_d["prog"], batch_d["mask"], batch_d["is_cls"])
                    pred = logits.float().argmax(dim=1).cpu().numpy()
                    tgt = batch_d["labels"].cpu().numpy()
                    correct += int((pred == tgt).sum())
                    total += len(tgt)
            val_acc = correct / max(1, total)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_path = f"{save_prefix}{fold}.pt"
        torch.save({"state": best_state or model.state_dict(), "classes": classes, "val_acc": best_val}, save_path)
        print(f"[T1-V2 KF] fold {fold+1}/{k_folds} saved {save_path} | val_acc={best_val:.4f}")
        ckpts.append({"state": best_state or model.state_dict(), "classes": classes, "val_acc": best_val})
    return ckpts


@torch.no_grad()
def _predict_task1_transformer_probs(ckpt: dict, paths: List[str], batch_size: int = 64) -> np.ndarray:
    device = _device()
    classes = ckpt["classes"]
    model = _MidiTransformerT1(len(classes)).to(device)
    state = ckpt["state"]
    if not isinstance(state, dict):
        state = state
    model.load_state_dict(state)
    model.eval()
    cache: dict = {}
    ds = _Task1Dataset(paths, None, cache)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _t1_collate(b, T1_MAX_LEN),
    )
    out = []
    for batch in tqdm(loader, desc="T1-V2 predict"):
        batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=_amp_enabled()):
            logits = model(
                batch_d["pitch"], batch_d["dur"], batch_d["ioi"],
                batch_d["vel"], batch_d["prog"], batch_d["mask"],
                batch_d["is_cls"],
            )
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)


def run1_v2(
    quick: bool = False,
    epochs: int = 35,
    ast_epochs: int = 15,
    transformer_ckpt: str = "task1_v2_transformer.pt",
    ast_ckpt: str = "task1_v2_ast.pt",
    hgb_weight: float = 0.40,
    tfm_weight: float = 0.20,
    ast_weight: float = 0.40,
    skip_transformer_train: bool = False,
    skip_ast_train: bool = False,
    skip_ast: bool = False,
    out_path: str | None = None,
) -> str:
    if out_path is None:
        out_path = "predictions1.json"
    """Train (or load) Transformer + MIDI-AST, train HGB, ensemble, write predictions."""
    print("\n=== Task 1 V2: HGB + MIDI Transformer + MIDI-AST ensemble ===")

    # 1) Transformer
    if skip_transformer_train and os.path.isfile(transformer_ckpt):
        print(f"Loading existing Task 1 Transformer: {transformer_ckpt}")
        tfm_ckpt = torch.load(transformer_ckpt, map_location="cpu", weights_only=False)
    else:
        tfm_ckpt = train_task1_transformer(
            epochs=(3 if quick else epochs), quick=quick, save_path=transformer_ckpt,
        )

    # 2) MIDI-AST (sine render -> AST fine-tune)
    ast_ckpt_loaded = None
    if not skip_ast:
        if skip_ast_train and os.path.isfile(ast_ckpt):
            print(f"Loading existing Task 1 AST: {ast_ckpt}")
            ast_ckpt_loaded = torch.load(ast_ckpt, map_location="cpu", weights_only=False)
        else:
            ast_ckpt_loaded = train_task1_ast(
                epochs=(3 if quick else ast_epochs), quick=quick, save_path=ast_ckpt,
            )

    # 3) HGB (use the original Model1 pipeline)
    hgb = Model1()
    hgb.train(os.path.join(DATAROOT1, "train.json"))

    # 4) Predict
    test_paths = _read_task1_test()
    tfm_probs = _predict_task1_transformer_probs(tfm_ckpt, test_paths)

    X = hgb._featurize_many(test_paths, "Task1 V2 HGB feats")
    hgb_probs = np.mean([m.predict_proba(X) for m in hgb.models], axis=0)
    hgb_classes = list(hgb.models[0].classes_)
    tfm_classes = tfm_ckpt["classes"]

    if ast_ckpt_loaded is not None:
        ast_probs = _predict_task1_ast_probs(ast_ckpt_loaded, test_paths, tta_shifts=(1 if quick else 3))
        ast_classes = ast_ckpt_loaded["classes"]
    else:
        ast_probs = None
        ast_classes = []

    all_classes = sorted(set(hgb_classes) | set(tfm_classes) | set(ast_classes))
    cidx = {c: i for i, c in enumerate(all_classes)}
    n_test = len(test_paths)
    K = len(all_classes)
    H = np.zeros((n_test, K), dtype=np.float32)
    T = np.zeros_like(H)
    A = np.zeros_like(H)
    for j, c in enumerate(hgb_classes):
        H[:, cidx[int(c)]] = hgb_probs[:, j]
    for j, c in enumerate(tfm_classes):
        T[:, cidx[int(c)]] = tfm_probs[:, j]
    if ast_probs is not None:
        for j, c in enumerate(ast_classes):
            A[:, cidx[int(c)]] = ast_probs[:, j]

    aw = ast_weight if ast_probs is not None else 0.0
    w = hgb_weight + tfm_weight + aw
    blend = (hgb_weight / w) * H + (tfm_weight / w) * T + (aw / w) * A
    pred_idx = np.argmax(blend, axis=1)
    pred_class = [all_classes[i] for i in pred_idx]

    predictions = {p: int(c) for p, c in zip(test_paths, pred_class)}
    for k in test_paths:
        if k in hgb.train_labels:
            predictions[k] = int(hgb.train_labels[k])
    write_submission_predictions(predictions, out_path)
    print(f"[T1-V2] wrote {out_path} (HGB w={hgb_weight}, Tfm w={tfm_weight}, AST w={aw})")
    return out_path


# ---------------------------------------------------------------------------
# Task 1 — MIDI -> sine audio -> AST classifier
# ---------------------------------------------------------------------------
T1_AST_SR = 16000
T1_AST_DURATION = 10.0
T1_AST_BATCH = int(os.environ.get("TASK1_AST_BATCH", "8"))
T1_AST_EVAL_BATCH = int(os.environ.get("TASK1_AST_EVAL_BATCH", "8"))
T1_AUDIO_CACHE_DIR = "cache_task1_audio"


_T1_HARMONICS = ((1.0, 1.0), (2.0, 0.45), (3.0, 0.22), (4.0, 0.10), (5.0, 0.05))


def _render_midi_to_wave(path: str, sr: int = T1_AST_SR, duration: float = T1_AST_DURATION) -> np.ndarray:
    """Pure-numpy piano-ish renderer: harmonics + ADSR with exponential decay."""
    midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT1, path))
    tpb = max(midi.ticks_per_beat, 1)
    tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
    sec_per_tick = 60.0 / (tempo * tpb)
    target_len = int(sr * duration)
    wave = np.zeros(target_len, dtype=np.float32)
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            t0 = n.start * sec_per_tick
            if t0 >= duration:
                continue
            t1 = min(n.end * sec_per_tick, duration)
            i0, i1 = int(t0 * sr), int(t1 * sr)
            length = i1 - i0
            if length <= 0:
                continue
            freq = 440.0 * (2.0 ** ((n.pitch - 69) / 12.0))
            t = np.arange(length, dtype=np.float32) / sr
            amp = (n.velocity / 127.0) * 0.22
            # Piano-like envelope: 5 ms attack, exponential decay over sustain, 30 ms release
            env = np.empty(length, dtype=np.float32)
            atk = min(int(0.005 * sr), max(1, length // 8))
            rel = min(int(0.03 * sr), max(1, length // 4))
            if atk > 0:
                env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float32)
            sustain_end = max(atk, length - rel)
            if sustain_end > atk:
                t_sust = np.arange(sustain_end - atk, dtype=np.float32) / sr
                # Faster decay for higher notes (piano-realistic)
                decay_rate = 1.5 + max(0.0, (n.pitch - 60) / 24.0)
                env[atk:sustain_end] = np.exp(-t_sust * decay_rate)
            if rel > 0 and sustain_end < length:
                last = env[sustain_end - 1] if sustain_end > 0 else 1.0
                env[sustain_end:] = np.linspace(last, 0.0, length - sustain_end, dtype=np.float32)
            # Harmonics: piano spectrum approximated
            signal = np.zeros(length, dtype=np.float32)
            for h_freq, h_amp in _T1_HARMONICS:
                # High harmonics fade if pitch is high (anti-alias-ish)
                if freq * h_freq > 0.45 * sr:
                    continue
                signal += h_amp * np.sin(2.0 * np.pi * freq * h_freq * t)
            wave[i0:i1] += amp * env * signal
    peak = float(np.max(np.abs(wave))) + 1e-9
    if peak > 1.0:
        wave = wave / peak
    return wave


def _t1_audio_cache_path(midi_path: str) -> str:
    os.makedirs(T1_AUDIO_CACHE_DIR, exist_ok=True)
    base = midi_path.replace("/", "__").replace("\\", "__").replace(".mid", ".npy")
    return os.path.join(T1_AUDIO_CACHE_DIR, base)


def _t1_get_wave(midi_path: str) -> np.ndarray:
    cache = _t1_audio_cache_path(midi_path)
    if os.path.isfile(cache):
        try:
            return np.load(cache)
        except Exception:
            pass
    wave = _render_midi_to_wave(midi_path)
    try:
        np.save(cache, wave.astype(np.float32))
    except Exception:
        pass
    return wave


class _Task1ASTDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int] | None, feat_extractor, train: bool = False):
        self.paths = paths
        self.labels = labels
        self.fe = feat_extractor
        self.train = train
        self.target_len = int(T1_AST_SR * T1_AST_DURATION)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        wav = _t1_get_wave(p)
        if len(wav) < self.target_len:
            wav = np.pad(wav, (0, self.target_len - len(wav)))
        else:
            wav = wav[: self.target_len]
        if self.train:
            if random.random() < 0.5:
                wav = wav * random.uniform(0.8, 1.25)
            if random.random() < 0.3:
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * 0.003
            if random.random() < 0.3:
                shift = random.randint(-int(0.2 * T1_AST_SR), int(0.2 * T1_AST_SR))
                wav = np.roll(wav, shift)
        y = -1 if self.labels is None else int(self.labels[idx])
        return wav.astype(np.float32), y, p


def _t1_ast_collate(fe):
    def _fn(batch):
        wavs = [b[0] for b in batch]
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        paths = [b[2] for b in batch]
        inputs = fe(wavs, sampling_rate=T1_AST_SR, return_tensors="pt")
        return {"input_values": inputs["input_values"], "labels": torch.from_numpy(ys), "paths": paths}
    return _fn


def _t1_ast_load_resources(n_classes: int):
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    feat = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        AST_MODEL_NAME,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )
    return feat, model


def train_task1_ast(
    epochs: int = 15,
    batch_size: int = T1_AST_BATCH,
    eval_batch_size: int = T1_AST_EVAL_BATCH,
    grad_accum: int = 4,
    lr_head: float = 5e-4,
    lr_backbone: float = 2e-5,
    weight_decay: float = 1e-2,
    val_ratio: float = 0.12,
    quick: bool = False,
    save_path: str = "task1_v2_ast.pt",
    freeze_backbone_epochs: int = 1,
) -> dict:
    """Fine-tune AST on sine-rendered MIDI for composer classification."""
    device = _device()
    paths, labels = _read_task1_train()
    classes = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_idx[c] for c in labels], dtype=np.int64)

    if quick:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(paths))[: min(256, len(paths))]
        paths = [paths[i] for i in idx]
        y_all = y_all[idx]
        epochs = min(epochs, 3)

    n = len(paths)
    val_n = max(8, int(n * val_ratio))
    tr_idx = np.arange(0, n - val_n)
    va_idx = np.arange(n - val_n, n)

    feat_extractor, model = _t1_ast_load_resources(len(classes))
    model = model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_ds = _Task1ASTDataset([paths[i] for i in tr_idx], [int(y_all[i]) for i in tr_idx], feat_extractor, train=True)
    val_ds = _Task1ASTDataset([paths[i] for i in va_idx], [int(y_all[i]) for i in va_idx], feat_extractor, train=False)
    collate = _t1_ast_collate(feat_extractor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    cls_count = np.bincount(y_all[tr_idx], minlength=len(classes)).astype(np.float32)
    cls_w = (cls_count.mean() / np.maximum(cls_count, 1.0)).clip(0.25, 4.0)
    weight = torch.tensor(cls_w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    def _split(named):
        head, body = [], []
        for name, p in named:
            if not p.requires_grad:
                continue
            (head if "classifier" in name else body).append(p)
        return head, body

    def _build_optim(freeze: bool):
        for name, p in model.named_parameters():
            if freeze and ("classifier" not in name):
                p.requires_grad = False
            else:
                p.requires_grad = True
        head, body = _split(model.named_parameters())
        groups = []
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        if body:
            groups.append({"params": body, "lr": lr_backbone, "weight_decay": weight_decay})
        return torch.optim.AdamW(groups)

    optim = _build_optim(freeze=True)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(50, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())

    best_acc = -1.0
    best_state = None
    step = 0
    unfrozen = False
    for epoch in range(epochs):
        if (not unfrozen) and epoch >= freeze_backbone_epochs:
            print(f"[T1-V2 AST] unfreezing backbone at epoch {epoch+1}")
            optim = _build_optim(freeze=False)
            sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
            unfrozen = True
        model.train()
        running, n_batches = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for bi, batch in enumerate(tqdm(train_loader, desc=f"T1-V2 AST epoch {epoch+1}/{epochs}")):
            input_values = _t3_specaug(batch["input_values"]).to(device, non_blocking=True)
            labels_t = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                out = model(input_values=input_values)
                loss = loss_fn(out.logits, labels_t) / grad_accum
            scaler.scale(loss).backward()
            n_batches += 1
            running += float(loss.detach()) * grad_accum
            if (bi + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    out = model(input_values=input_values)
                pred = out.logits.float().argmax(dim=1).cpu().numpy()
                tgt = batch["labels"].numpy()
                correct += int((pred == tgt).sum())
                total += len(tgt)
        val_acc = correct / max(1, total)
        print(f"[T1-V2 AST] epoch {epoch+1} loss={running/max(1,n_batches):.4f} val_tail_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state": best_state or model.state_dict(), "classes": classes, "val_tail_acc": best_acc}, save_path)
    print(f"[T1-V2 AST] saved {save_path} | best tail acc={best_acc:.4f}")
    return {"state": best_state or model.state_dict(), "classes": classes, "val_tail_acc": best_acc}


def train_task1_ast_kfold(
    k_folds: int = 3,
    epochs: int = 15,
    batch_size: int = T1_AST_BATCH,
    eval_batch_size: int = T1_AST_EVAL_BATCH,
    grad_accum: int = 4,
    lr_head: float = 5e-4,
    lr_backbone: float = 2e-5,
    weight_decay: float = 1e-2,
    quick: bool = False,
    save_prefix: str = "task1_v2_ast_fold",
    freeze_backbone_epochs: int = 1,
) -> List[dict]:
    """Train K-fold AST ensemble. Returns list of ckpts."""
    device = _device()
    paths, labels = _read_task1_train()
    classes = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_idx[c] for c in labels], dtype=np.int64)

    if quick:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(paths))[: min(256, len(paths))]
        paths = [paths[i] for i in idx]
        y_all = y_all[idx]
        epochs = min(epochs, 3)

    n = len(paths)
    rng = np.random.RandomState(SEED + 7)
    perm = rng.permutation(n)
    fold_size = n // k_folds
    ckpts = []

    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n
        va_idx = perm[val_start:val_end]
        tr_idx = np.concatenate([perm[:val_start], perm[val_end:]])

        feat_extractor, model = _t1_ast_load_resources(len(classes))
        model = model.to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        torch.manual_seed(SEED + 100 * fold)

        train_ds = _Task1ASTDataset([paths[i] for i in tr_idx], [int(y_all[i]) for i in tr_idx], feat_extractor, train=True)
        val_ds = _Task1ASTDataset([paths[i] for i in va_idx], [int(y_all[i]) for i in va_idx], feat_extractor, train=False)
        collate = _t1_ast_collate(feat_extractor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate)

        cls_count = np.bincount(y_all[tr_idx], minlength=len(classes)).astype(np.float32)
        cls_w = (cls_count.mean() / np.maximum(cls_count, 1.0)).clip(0.25, 4.0)
        weight = torch.tensor(cls_w, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

        def _split(named):
            head, body = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                (head if "classifier" in name else body).append(p)
            return head, body

        def _build_optim(freeze: bool):
            for name, p in model.named_parameters():
                if freeze and ("classifier" not in name):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            head, body = _split(model.named_parameters())
            groups = []
            if head:
                groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
            if body:
                groups.append({"params": body, "lr": lr_backbone, "weight_decay": weight_decay})
            return torch.optim.AdamW(groups)

        optim = _build_optim(freeze=True)
        steps_per_epoch = max(1, len(train_loader) // grad_accum)
        total_steps = steps_per_epoch * epochs
        warmup = max(50, total_steps // 20)

        def lr_lambda(step, _warmup=warmup, _total=total_steps):
            if step < _warmup:
                return step / max(1, _warmup)
            prog = (step - _warmup) / max(1, _total - _warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())
        best_acc, best_state = -1.0, None
        unfrozen = False
        for epoch in range(epochs):
            if (not unfrozen) and epoch >= freeze_backbone_epochs:
                optim = _build_optim(freeze=False)
                sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
                unfrozen = True
            model.train()
            running, n_batches = 0.0, 0
            optim.zero_grad(set_to_none=True)
            for bi, batch in enumerate(tqdm(train_loader, desc=f"T1-V2 AST KF{fold+1}/{k_folds} ep{epoch+1}/{epochs}")):
                input_values = _t3_specaug(batch["input_values"]).to(device, non_blocking=True)
                labels_t = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    out = model(input_values=input_values)
                    loss = loss_fn(out.logits, labels_t) / grad_accum
                scaler.scale(loss).backward()
                n_batches += 1
                running += float(loss.detach()) * grad_accum
                if (bi + 1) % grad_accum == 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_values = batch["input_values"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                        out = model(input_values=input_values)
                    pred = out.logits.float().argmax(dim=1).cpu().numpy()
                    tgt = batch["labels"].numpy()
                    correct += int((pred == tgt).sum())
                    total += len(tgt)
            val_acc = correct / max(1, total)
            print(f"[T1-V2 AST KF{fold+1}] epoch {epoch+1} loss={running/max(1,n_batches):.4f} val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_path = f"{save_prefix}{fold}.pt"
        torch.save({"state": best_state or model.state_dict(), "classes": classes, "val_acc": best_acc}, save_path)
        print(f"[T1-V2 AST KF] fold {fold+1}/{k_folds} saved {save_path} | val_acc={best_acc:.4f}")
        ckpts.append({"state": best_state or model.state_dict(), "classes": classes, "val_acc": best_acc})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ckpts


@torch.no_grad()
def _predict_task1_ast_probs(ckpt: dict, paths: List[str], batch_size: int = T1_AST_EVAL_BATCH, tta_shifts: int = 3) -> np.ndarray:
    device = _device()
    classes = ckpt["classes"]
    feat_extractor, model = _t1_ast_load_resources(len(classes))
    model = model.to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    ds = _Task1ASTDataset(paths, None, feat_extractor, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_t1_ast_collate(feat_extractor))
    base_probs = []
    base_paths = []
    for batch in tqdm(loader, desc="T1-V2 AST predict"):
        input_values = batch["input_values"].to(device, non_blocking=True)
        accum = None
        # Per-batch TTA: shift the feature-level time axis by small amounts.
        shifts = [0]
        if tta_shifts > 1 and input_values.dim() == 3:
            step = max(1, input_values.shape[1] // 20)
            for k in range(1, tta_shifts):
                shifts.append((k - tta_shifts // 2) * step)
        for sh in shifts:
            iv = input_values if sh == 0 else torch.roll(input_values, shifts=sh, dims=1)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                out = model(input_values=iv)
            p = F.softmax(out.logits.float(), dim=1).cpu().numpy()
            accum = p if accum is None else accum + p
        base_probs.append(accum / len(shifts))
        base_paths.extend(batch["paths"])
    probs = np.concatenate(base_probs, axis=0)
    # Re-order to match `paths`
    idx = {p: i for i, p in enumerate(base_paths)}
    out = np.zeros_like(probs)
    for i, p in enumerate(paths):
        out[i] = probs[idx[p]]
    return out


def run1_v2_kfold(
    quick: bool = False,
    k_folds: int = 3,
    epochs: int = 35,
    ast_epochs: int = 15,
    transformer_prefix: str = "task1_v2_transformer_fold",
    ast_prefix: str = "task1_v2_ast_fold",
    hgb_seeds: Tuple[int, ...] = (0, 17, 42),
    hgb_weight: float = 0.30,
    tfm_weight: float = 0.30,
    ast_weight: float = 0.40,
    skip_transformer_train: bool = False,
    skip_ast_train: bool = False,
    skip_ast: bool = False,
    out_path: str | None = None,
) -> str:
    """K-fold CV ensemble for T1: K-Transformer + K-AST + multi-seed HGB."""
    if out_path is None:
        out_path = "predictions1.json"
    print(f"\n=== Task 1 V2 K-FOLD: HGB(x{len(hgb_seeds)}) + Tfm(K={k_folds}) + AST(K={k_folds}) ===")

    # 1) Transformer K-fold
    tfm_ckpts: List[dict] = []
    if skip_transformer_train:
        for fold in range(k_folds):
            path = f"{transformer_prefix}{fold}.pt"
            if os.path.isfile(path):
                tfm_ckpts.append(torch.load(path, map_location="cpu", weights_only=False))
            else:
                print(f"[T1-V2 KF] missing {path}; training all folds from scratch")
                tfm_ckpts = []
                break
    if not tfm_ckpts:
        tfm_ckpts = train_task1_transformer_kfold(
            k_folds=k_folds, epochs=(3 if quick else epochs), quick=quick, save_prefix=transformer_prefix,
        )

    # 2) AST K-fold
    ast_ckpts: List[dict] = []
    if not skip_ast:
        if skip_ast_train:
            for fold in range(k_folds):
                path = f"{ast_prefix}{fold}.pt"
                if os.path.isfile(path):
                    ast_ckpts.append(torch.load(path, map_location="cpu", weights_only=False))
                else:
                    print(f"[T1-V2 AST KF] missing {path}; training all folds from scratch")
                    ast_ckpts = []
                    break
        if not ast_ckpts:
            ast_ckpts = train_task1_ast_kfold(
                k_folds=k_folds, epochs=(3 if quick else ast_epochs), quick=quick, save_prefix=ast_prefix,
            )

    # 3) HGB multi-seed: monkey-patch baseline SEED, train Model1 once per seed.
    import baseline as a1mod
    orig_seed = a1mod.SEED
    hgb_models = []
    hgb_classes_ref = None
    for seed in hgb_seeds:
        a1mod.SEED = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        m = Model1()
        m.train(os.path.join(DATAROOT1, "train.json"))
        hgb_models.append(m)
        if hgb_classes_ref is None:
            hgb_classes_ref = list(m.models[0].classes_)
    a1mod.SEED = orig_seed

    # 4) Predict
    test_paths = _read_task1_test()
    # Transformer probs: mean across folds
    tfm_probs_acc = None
    tfm_classes = tfm_ckpts[0]["classes"]
    for ckpt in tfm_ckpts:
        p = _predict_task1_transformer_probs(ckpt, test_paths)
        tfm_probs_acc = p if tfm_probs_acc is None else tfm_probs_acc + p
    tfm_probs = tfm_probs_acc / len(tfm_ckpts)

    # AST probs: mean across folds
    ast_probs = None
    ast_classes = []
    if ast_ckpts:
        ast_classes = ast_ckpts[0]["classes"]
        ast_acc = None
        for ckpt in ast_ckpts:
            p = _predict_task1_ast_probs(ckpt, test_paths, tta_shifts=(1 if quick else 3))
            ast_acc = p if ast_acc is None else ast_acc + p
        ast_probs = ast_acc / len(ast_ckpts)

    # HGB probs: mean across seeds
    X = hgb_models[0]._featurize_many(test_paths, "Task1 V2 HGB feats")
    hgb_probs_list = []
    for m in hgb_models:
        hgb_probs_list.append(np.mean([est.predict_proba(X) for est in m.models], axis=0))
    hgb_probs = np.mean(hgb_probs_list, axis=0)

    # Unified class index
    all_classes = sorted(set(hgb_classes_ref) | set(tfm_classes) | set(ast_classes))
    cidx = {c: i for i, c in enumerate(all_classes)}
    K = len(all_classes)
    H = np.zeros((len(test_paths), K), dtype=np.float32)
    Tm = np.zeros_like(H)
    A = np.zeros_like(H)
    for j, c in enumerate(hgb_classes_ref):
        H[:, cidx[int(c)]] = hgb_probs[:, j]
    for j, c in enumerate(tfm_classes):
        Tm[:, cidx[int(c)]] = tfm_probs[:, j]
    if ast_probs is not None:
        for j, c in enumerate(ast_classes):
            A[:, cidx[int(c)]] = ast_probs[:, j]

    aw = ast_weight if ast_probs is not None else 0.0
    w = hgb_weight + tfm_weight + aw
    blend = (hgb_weight / w) * H + (tfm_weight / w) * Tm + (aw / w) * A
    pred_idx = np.argmax(blend, axis=1)
    pred_class = [all_classes[i] for i in pred_idx]

    predictions = {p: int(c) for p, c in zip(test_paths, pred_class)}
    # Train-leak fix from baseline
    base_labels = hgb_models[0].train_labels
    for k in test_paths:
        if k in base_labels:
            predictions[k] = int(base_labels[k])
    write_submission_predictions(predictions, out_path)
    print(f"[T1-V2 KF] wrote {out_path} (HGB w={hgb_weight}, Tfm w={tfm_weight}, AST w={aw}; K={k_folds}, HGB seeds={hgb_seeds})")
    return out_path


# ===========================================================================
# Task 2 — Pair Transformer
# ===========================================================================
T2_MAX_PER_SEG = 192                # notes per segment
T2_DUR_BINS = 16
T2_IOI_BINS = 16
T2_VEL_BINS = 8
T2_PROG_BINS = 16
T2_PITCH_OFFSET = 4                 # 0=PAD, 1=CLS, 2=SEPa, 3=SEPb
T2_PAD_ID = 0
T2_CLS_ID = 1
T2_SEPA_ID = 2
T2_SEPB_ID = 3

T2_DUR_LO, T2_DUR_HI = math.log(0.05), math.log(8.0)
T2_IOI_LO, T2_IOI_HI = math.log(0.02), math.log(4.0)

T2_D_MODEL = 192
T2_N_HEADS = 6
T2_N_LAYERS = 4
T2_FFN = 512
T2_DROPOUT = 0.25


def _tokenize_midi_task2(path: str) -> dict:
    midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT2, path))
    tpb = max(midi.ticks_per_beat, 1)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        prog = int(getattr(inst, "program", 0)) // 8
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity, prog))
    if not notes:
        return None
    notes.sort(key=lambda t: (t[0], t[2]))
    n = len(notes)
    pitch = np.zeros(n, dtype=np.int64)
    dur = np.zeros(n, dtype=np.int64)
    ioi = np.zeros(n, dtype=np.int64)
    vel = np.zeros(n, dtype=np.int64)
    prog = np.zeros(n, dtype=np.int64)
    last = None
    for i, (s, e, p, v, pr) in enumerate(notes):
        dur_b = max((e - s) / tpb, 1e-3)
        ioi_b = 0.0 if last is None else max((s - last) / tpb, 0.0)
        last = s
        pitch[i] = int(p) + T2_PITCH_OFFSET
        dur[i] = _log_bin(dur_b, T2_DUR_LO, T2_DUR_HI, T2_DUR_BINS)
        ioi[i] = _log_bin(max(ioi_b, 1e-3), T2_IOI_LO, T2_IOI_HI, T2_IOI_BINS)
        vel[i] = int(min(7, max(0, v // 16)))
        prog[i] = int(min(T2_PROG_BINS - 1, max(0, pr)))
    return {"pitch": pitch, "dur": dur, "ioi": ioi, "vel": vel, "prog": prog, "n": n}


def _seg_take(seg: dict, max_n: int, train: bool):
    n = seg["n"]
    if n <= max_n:
        return slice(0, n)
    if train:
        start = np.random.randint(0, n - max_n + 1)
    else:
        start = (n - max_n) // 2
    return slice(start, start + max_n)


def _t2_collate(batch: list, max_per_seg: int = T2_MAX_PER_SEG, train: bool = True):
    """Build sequence [CLS, SEPa, A notes, SEPb, B notes]."""
    B = len(batch)
    L = 1 + 1 + max_per_seg + 1 + max_per_seg
    pitch = np.zeros((B, L), dtype=np.int64)
    dur = np.zeros((B, L), dtype=np.int64)
    ioi = np.zeros((B, L), dtype=np.int64)
    vel = np.zeros((B, L), dtype=np.int64)
    prog = np.zeros((B, L), dtype=np.int64)
    seg_id = np.zeros((B, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    labels = np.zeros(B, dtype=np.float32)

    for i, (sa, sb, y) in enumerate(batch):
        pitch[i, 0] = T2_CLS_ID
        pitch[i, 1] = T2_SEPA_ID
        mask[i, 0:2] = True
        seg_id[i, 1] = 1
        pos_a = 2
        if sa is not None:
            sl = _seg_take(sa, max_per_seg, train)
            take = sl.stop - sl.start
            pitch[i, pos_a:pos_a + take] = sa["pitch"][sl]
            dur[i, pos_a:pos_a + take] = sa["dur"][sl]
            ioi[i, pos_a:pos_a + take] = sa["ioi"][sl]
            vel[i, pos_a:pos_a + take] = sa["vel"][sl]
            prog[i, pos_a:pos_a + take] = sa["prog"][sl]
            seg_id[i, pos_a:pos_a + take] = 1
            mask[i, pos_a:pos_a + take] = True
        pos_sb = 2 + max_per_seg
        pitch[i, pos_sb] = T2_SEPB_ID
        seg_id[i, pos_sb] = 2
        mask[i, pos_sb] = True
        pos_b = pos_sb + 1
        if sb is not None:
            sl = _seg_take(sb, max_per_seg, train)
            take = sl.stop - sl.start
            pitch[i, pos_b:pos_b + take] = sb["pitch"][sl]
            dur[i, pos_b:pos_b + take] = sb["dur"][sl]
            ioi[i, pos_b:pos_b + take] = sb["ioi"][sl]
            vel[i, pos_b:pos_b + take] = sb["vel"][sl]
            prog[i, pos_b:pos_b + take] = sb["prog"][sl]
            seg_id[i, pos_b:pos_b + take] = 2
            mask[i, pos_b:pos_b + take] = True
        labels[i] = float(y) if y is not None else 0.0
    return {
        "pitch": torch.from_numpy(pitch),
        "dur": torch.from_numpy(dur),
        "ioi": torch.from_numpy(ioi),
        "vel": torch.from_numpy(vel),
        "prog": torch.from_numpy(prog),
        "seg_id": torch.from_numpy(seg_id),
        "mask": torch.from_numpy(mask),
        "labels": torch.from_numpy(labels),
    }


class _Task2Dataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[bool] | None, cache: dict, augment_swap: bool = False):
        self.pairs = pairs
        self.labels = labels
        self.cache = cache
        self.augment_swap = augment_swap  # synthetic flip during training

    def _bundle(self, p):
        if p not in self.cache:
            self.cache[p] = _tokenize_midi_task2(p)
        return self.cache[p]

    def __len__(self):
        n = len(self.pairs)
        return n * (2 if self.augment_swap else 1)

    def __getitem__(self, idx):
        if self.augment_swap:
            real_idx = idx // 2
            swap = bool(idx % 2)
        else:
            real_idx = idx
            swap = False
        p1, p2 = self.pairs[real_idx]
        sa = self._bundle(p1)
        sb = self._bundle(p2)
        y = None if self.labels is None else bool(self.labels[real_idx])
        if swap:
            sa, sb = sb, sa
            y = (not y) if y is not None else None
        return sa, sb, (1.0 if y else 0.0) if y is not None else 0.0


class _PairTransformerT2(nn.Module):
    def __init__(self):
        super().__init__()
        d = T2_D_MODEL
        n_pitch_tokens = 128 + T2_PITCH_OFFSET
        self.pitch_emb = nn.Embedding(n_pitch_tokens, d, padding_idx=T2_PAD_ID)
        self.dur_emb = nn.Embedding(T2_DUR_BINS, d)
        self.ioi_emb = nn.Embedding(T2_IOI_BINS, d)
        self.vel_emb = nn.Embedding(T2_VEL_BINS, d)
        self.prog_emb = nn.Embedding(T2_PROG_BINS, d)
        self.seg_emb = nn.Embedding(3, d)
        self.pos_emb = nn.Embedding(1024, d)
        self.input_norm = nn.LayerNorm(d)
        self.input_drop = nn.Dropout(T2_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=T2_N_HEADS, dim_feedforward=T2_FFN,
            dropout=T2_DROPOUT, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=T2_N_LAYERS)
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Dropout(T2_DROPOUT),
            nn.Linear(d, 1),
        )

    def forward(self, pitch, dur, ioi, vel, prog, seg_id, mask):
        B, L = pitch.shape
        pos = torch.arange(L, device=pitch.device).unsqueeze(0).expand(B, L)
        x = (
            self.pitch_emb(pitch)
            + self.dur_emb(dur)
            + self.ioi_emb(ioi)
            + self.vel_emb(vel)
            + self.prog_emb(prog)
            + self.seg_emb(seg_id)
            + self.pos_emb(pos)
        )
        x = self.input_drop(self.input_norm(x))
        x = self.encoder(x, src_key_padding_mask=~mask)
        cls = x[:, 0, :]
        m = mask.float().unsqueeze(-1)
        msum = m.sum(dim=1).clamp(min=1.0)
        avg = (x * m).sum(dim=1) / msum
        return self.head(self.head_norm(cls + avg)).squeeze(-1)


def _read_task2_train() -> Tuple[List[Tuple[str, str]], List[bool]]:
    with open(os.path.join(DATAROOT2, "train.json")) as f:
        d = eval(f.read())
    pairs = list(d.keys())
    labels = [bool(d[k]) for k in pairs]
    return pairs, labels


def _read_task2_test() -> List[Tuple[str, str]]:
    with open(os.path.join(DATAROOT2, "test.json")) as f:
        d = eval(f.read())
    return list(d.keys()) if isinstance(d, dict) else list(d)


def train_task2_transformer(
    epochs: int = 18,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 0.02,
    val_ratio: float = 0.12,
    quick: bool = False,
    save_path: str = "task2_v2_transformer.pt",
) -> dict:
    device = _device()
    pairs, labels = _read_task2_train()
    if quick:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(pairs))[: min(512, len(pairs))]
        pairs = [pairs[i] for i in idx]
        labels = [labels[i] for i in idx]
        epochs = min(epochs, 4)

    n = len(pairs)
    rng = np.random.RandomState(SEED + 1)
    perm = rng.permutation(n)
    val_n = max(8, int(n * val_ratio))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    cache: dict = {}
    train_ds = _Task2Dataset(
        [pairs[i] for i in tr_idx], [labels[i] for i in tr_idx], cache, augment_swap=True
    )
    val_ds = _Task2Dataset(
        [pairs[i] for i in val_idx], [labels[i] for i in val_idx], cache, augment_swap=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: _t2_collate(b, T2_MAX_PER_SEG, train=True),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: _t2_collate(b, T2_MAX_PER_SEG, train=False),
    )

    model = _PairTransformerT2().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    warmup = max(50, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"T2-V2 epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                logits = model(
                    batch["pitch"], batch["dur"], batch["ioi"], batch["vel"],
                    batch["prog"], batch["seg_id"], batch["mask"],
                )
                loss = loss_fn(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            step += 1

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                # Forward
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    la = model(
                        batch_d["pitch"], batch_d["dur"], batch_d["ioi"], batch_d["vel"],
                        batch_d["prog"], batch_d["seg_id"], batch_d["mask"],
                    )
                pa = torch.sigmoid(la.float()).cpu().numpy()
                pred = (pa >= 0.5).astype(np.int32)
                tgt = batch_d["labels"].cpu().numpy().astype(np.int32)
                correct += int((pred == tgt).sum())
                total += len(tgt)
        val_acc = correct / max(1, total)
        print(f"[T2-V2] epoch {epoch+1} loss={running/max(1,len(train_loader)):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state": best_state or model.state_dict(), "val_acc": best_val}, save_path)
    print(f"[T2-V2] saved {save_path} | best val acc={best_val:.4f}")
    return {"state": best_state or model.state_dict(), "val_acc": best_val}


@torch.no_grad()
def _predict_task2_transformer_probs(
    ckpt: dict, pairs: List[Tuple[str, str]], batch_size: int = 24, average_swap: bool = True,
) -> np.ndarray:
    """Return P(A then B) for each pair, using forward+reverse averaging."""
    device = _device()
    model = _PairTransformerT2().to(device)
    state = ckpt["state"]
    model.load_state_dict(state)
    model.eval()
    cache: dict = {}

    def _probs(ordered_pairs):
        ds = _Task2Dataset(ordered_pairs, None, cache, augment_swap=False)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=lambda b: _t2_collate(b, T2_MAX_PER_SEG, train=False),
        )
        out = []
        for batch in tqdm(loader, desc="T2-V2 predict"):
            batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                lg = model(
                    batch_d["pitch"], batch_d["dur"], batch_d["ioi"], batch_d["vel"],
                    batch_d["prog"], batch_d["seg_id"], batch_d["mask"],
                )
            out.append(torch.sigmoid(lg.float()).cpu().numpy())
        return np.concatenate(out, axis=0)

    pa = _probs(pairs)
    if not average_swap:
        return pa
    pb = _probs([(b, a) for (a, b) in pairs])
    eps = 1e-6
    pa = np.clip(pa, eps, 1.0 - eps)
    pb = np.clip(pb, eps, 1.0 - eps)
    la = np.log(pa / (1.0 - pa))
    lb_swap = np.log((1.0 - pb) / pb)
    return 1.0 / (1.0 + np.exp(-0.5 * (la + lb_swap)))


def run2_v2(
    quick: bool = False,
    epochs: int = 18,
    transformer_ckpt: str = "task2_v2_transformer.pt",
    tree_weight: float = 0.5,
    tfm_weight: float = 0.5,
    skip_transformer_train: bool = False,
    skip_tree_train: bool = False,
    out_path: str | None = None,
) -> str:
    """Train Task 2 Transformer + tree ensemble, average and write predictions."""
    if out_path is None:
        out_path = "predictions2.json"
    print("\n=== Task 2 V2: Tree ensemble + Pair Transformer ===")

    if skip_transformer_train and os.path.isfile(transformer_ckpt):
        ckpt = torch.load(transformer_ckpt, map_location="cpu", weights_only=False)
    else:
        ckpt = train_task2_transformer(
            epochs=(3 if quick else epochs), quick=quick, save_path=transformer_ckpt,
        )

    tree = Model2()
    if not skip_tree_train:
        tree.train(os.path.join(DATAROOT2, "train.json"))

    test_pairs = _read_task2_test()

    tfm_p = _predict_task2_transformer_probs(ckpt, test_pairs)

    if not skip_tree_train:
        # Mirror Model2.predict probability blending
        Xa = np.stack([tree._pair_vec(p1, p2) for (p1, p2) in tqdm(test_pairs, desc="T2-V2 tree feats (a)")])
        Xb = np.stack([tree._pair_vec(p2, p1) for (p1, p2) in tqdm(test_pairs, desc="T2-V2 tree feats (b)")])
        models = tree.models or [tree.model]
        proba_a = np.mean([m.predict_proba(Xa)[:, 1] for m in models], axis=0)
        proba_b = np.mean([m.predict_proba(Xb)[:, 1] for m in models], axis=0)
        eps = 1e-6
        pa = np.clip(proba_a, eps, 1.0 - eps)
        pb = np.clip(proba_b, eps, 1.0 - eps)
        la = np.log(pa / (1.0 - pa))
        lb_swap = np.log((1.0 - pb) / pb)
        tree_p = 1.0 / (1.0 + np.exp(-0.5 * (la + lb_swap)))
        w = tree_weight + tfm_weight
        combined = (tree_weight / w) * tree_p + (tfm_weight / w) * tfm_p
    else:
        combined = tfm_p

    predictions = {k: bool(c >= 0.5) for k, c in zip(test_pairs, combined)}
    write_submission_predictions(predictions, out_path)
    print(f"[T2-V2] wrote {out_path}")
    return out_path


# ===========================================================================
# Task 3 — AST fine-tune (HuggingFace, multilabel)
# ===========================================================================
AST_MODEL_NAME = os.environ.get(
    "TASK3_AST_NAME", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
AST_SAMPLE_RATE = 16000
AST_DURATION = 10.0  # seconds — must match training
AST_BATCH_SIZE = int(os.environ.get("TASK3_AST_BATCH", "8"))
AST_EVAL_BATCH = int(os.environ.get("TASK3_AST_EVAL_BATCH", "8"))


def _ast_load_resources():
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    feat = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    # Multilabel — replace classifier head.
    model = ASTForAudioClassification.from_pretrained(
        AST_MODEL_NAME,
        num_labels=len(TAGS),
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification",
    )
    return feat, model


class _ASTDataset(Dataset):
    def __init__(self, meta: dict, train: bool, feat_extractor):
        import librosa  # imported lazily
        self.librosa = librosa
        self.meta = meta
        self.keys = list(meta.keys())
        self.train = train
        self.fe = feat_extractor
        self.cache: dict = {}
        self.target_len = int(AST_SAMPLE_RATE * AST_DURATION)

    def __len__(self):
        return len(self.keys)

    def _load_wave(self, p):
        if p in self.cache:
            return self.cache[p].copy()
        wav, _ = self.librosa.load(
            os.path.join(DATAROOT3, p), sr=AST_SAMPLE_RATE, mono=True
        )
        if len(wav) < self.target_len:
            wav = np.pad(wav, (0, self.target_len - len(wav)))
        else:
            wav = wav[: self.target_len]
        self.cache[p] = wav.astype(np.float32)
        return wav.copy()

    def __getitem__(self, idx):
        p = self.keys[idx]
        wav = self._load_wave(p)
        if self.train:
            # Light wave augmentation: random gain + tiny pitch/time
            if random.random() < 0.5:
                wav = wav * random.uniform(0.8, 1.25)
            if random.random() < 0.25:
                wav = self.librosa.effects.pitch_shift(
                    wav, sr=AST_SAMPLE_RATE, n_steps=random.randint(-2, 2)
                )
                if len(wav) < self.target_len:
                    wav = np.pad(wav, (0, self.target_len - len(wav)))
                else:
                    wav = wav[: self.target_len]
            if random.random() < 0.25:
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * 0.003
        tags = self.meta[p]
        label = np.array([1.0 if t in tags else 0.0 for t in TAGS], dtype=np.float32)
        return wav, label, p


def _ast_collate_fn(fe):
    def _fn(batch):
        wavs = [b[0] for b in batch]
        labels = np.stack([b[1] for b in batch])
        paths = [b[2] for b in batch]
        inputs = fe(wavs, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        return {
            "input_values": inputs["input_values"],
            "labels": torch.from_numpy(labels),
            "paths": paths,
        }
    return _fn


def _read_task3_train() -> dict:
    with open(os.path.join(DATAROOT3, "train.json")) as f:
        return eval(f.read())


def _read_task3_test() -> List[str]:
    with open(os.path.join(DATAROOT3, "test.json")) as f:
        d = eval(f.read())
    return list(d) if not isinstance(d, dict) else list(d.keys())


def _t3_specaug(input_values: torch.Tensor) -> torch.Tensor:
    """SpecAugment on AST inputs (B, T, F) — masks time and freq bands."""
    if input_values.dim() != 3:
        return input_values
    x = input_values.clone()
    B, T, Fdim = x.shape
    for i in range(B):
        if random.random() < 0.5:
            t_mask = random.randint(0, max(1, T // 12))
            t0 = random.randint(0, max(0, T - t_mask))
            x[i, t0:t0 + t_mask, :] = 0.0
        if random.random() < 0.5:
            f_mask = random.randint(0, max(1, Fdim // 12))
            f0 = random.randint(0, max(0, Fdim - f_mask))
            x[i, :, f0:f0 + f_mask] = 0.0
    return x


def train_task3_ast(
    epochs: int = 12,
    batch_size: int = AST_BATCH_SIZE,
    eval_batch_size: int = AST_EVAL_BATCH,
    grad_accum: int = 4,
    lr_head: float = 5e-4,
    lr_backbone: float = 2e-5,
    weight_decay: float = 1e-2,
    val_ratio: float = 0.1,
    quick: bool = False,
    save_path: str = "task3_v2_ast.pt",
    max_train: int | None = None,
    freeze_backbone_epochs: int = 1,
) -> dict:
    """Fine-tune AST. Returns dict with model state, mean/std passthrough (AST FE does its own normalization)."""
    from sklearn.metrics import average_precision_score

    device = _device()
    feat_extractor, model = _ast_load_resources()
    model = model.to(device)

    # Optional gradient checkpointing to fit 8 GB
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    meta = _read_task3_train()
    keys = list(meta.keys())
    rng = random.Random(SEED)
    rng.shuffle(keys)
    if quick:
        keys = keys[: min(512, len(keys))]
        epochs = min(epochs, 3)
    elif max_train is not None and max_train > 0:
        keys = keys[:max_train]
    n = len(keys)
    val_n = max(16, int(n * val_ratio))
    val_keys = keys[:val_n]
    tr_keys = keys[val_n:]
    train_meta = {k: meta[k] for k in tr_keys}
    val_meta = {k: meta[k] for k in val_keys}

    train_ds = _ASTDataset(train_meta, train=True, feat_extractor=feat_extractor)
    val_ds = _ASTDataset(val_meta, train=False, feat_extractor=feat_extractor)
    collate = _ast_collate_fn(feat_extractor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    def _split_params(named):
        head, body = [], []
        for name, p in named:
            if not p.requires_grad:
                continue
            if "classifier" in name:
                head.append(p)
            else:
                body.append(p)
        return head, body

    def _build_optim(freeze: bool):
        for name, p in model.named_parameters():
            if freeze and ("classifier" not in name):
                p.requires_grad = False
            else:
                p.requires_grad = True
        head, body = _split_params(model.named_parameters())
        groups = []
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        if body:
            groups.append({"params": body, "lr": lr_backbone, "weight_decay": weight_decay})
        return torch.optim.AdamW(groups)

    optim = _build_optim(freeze=True)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(50, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled())
    loss_fn = nn.BCEWithLogitsLoss()

    best_map = -1.0
    best_state = None
    step = 0
    unfrozen = False
    for epoch in range(epochs):
        if (not unfrozen) and epoch >= freeze_backbone_epochs:
            print(f"[T3-V2 AST] unfreezing backbone at epoch {epoch+1}")
            optim = _build_optim(freeze=False)
            sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
            unfrozen = True

        model.train()
        running = 0.0
        n_batches = 0
        optim.zero_grad(set_to_none=True)
        for bi, batch in enumerate(tqdm(train_loader, desc=f"T3-V2 AST epoch {epoch+1}/{epochs}")):
            input_values = _t3_specaug(batch["input_values"]).to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                out = model(input_values=input_values)
                loss = loss_fn(out.logits, labels) / grad_accum
            scaler.scale(loss).backward()
            n_batches += 1
            running += float(loss.detach()) * grad_accum
            if (bi + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1

        # Eval
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device, non_blocking=True)
                labels = batch["labels"]
                with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                    out = model(input_values=input_values)
                p = torch.sigmoid(out.logits.float()).cpu().numpy()
                all_p.append(p)
                all_y.append(labels.numpy())
        P = np.concatenate(all_p)
        Y = np.concatenate(all_y)
        try:
            val_map = float(average_precision_score(Y, P, average="macro"))
        except ValueError:
            val_map = float("nan")
        print(f"[T3-V2 AST] epoch {epoch+1} loss={running/max(1,n_batches):.4f} val_mAP={val_map:.4f}")
        if math.isfinite(val_map) and val_map > best_map:
            best_map = val_map
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state": best_state or model.state_dict(), "val_map": best_map}, save_path)
    print(f"[T3-V2 AST] saved {save_path} | best val mAP={best_map:.4f}")
    return {"state": best_state or model.state_dict(), "val_map": best_map}


@torch.no_grad()
def _predict_task3_ast_probs(ckpt: dict, test_keys: List[str], batch_size: int = AST_EVAL_BATCH, tta_shifts: int = 3) -> Tuple[np.ndarray, List[str]]:
    device = _device()
    feat_extractor, model = _ast_load_resources()
    model = model.to(device)
    state = ckpt["state"]
    model.load_state_dict(state)
    model.eval()
    meta = {k: [] for k in test_keys}
    ds = _ASTDataset(meta, train=False, feat_extractor=feat_extractor)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_ast_collate_fn(feat_extractor),
    )
    probs, paths = [], []
    for batch in tqdm(loader, desc="T3-V2 AST predict"):
        input_values = batch["input_values"].to(device, non_blocking=True)
        shifts = [0]
        if tta_shifts > 1 and input_values.dim() == 3:
            step = max(1, input_values.shape[1] // 20)
            for k in range(1, tta_shifts):
                shifts.append((k - tta_shifts // 2) * step)
        accum = None
        for sh in shifts:
            iv = input_values if sh == 0 else torch.roll(input_values, shifts=sh, dims=1)
            with torch.amp.autocast("cuda", enabled=_amp_enabled()):
                out = model(input_values=iv)
            p = torch.sigmoid(out.logits.float()).cpu().numpy()
            accum = p if accum is None else accum + p
        probs.append(accum / len(shifts))
        paths.extend(batch["paths"])
    return np.concatenate(probs), paths


@torch.no_grad()
def _predict_task3_resnet_probs(test_keys: List[str], tta_shifts: int = 5) -> Tuple[np.ndarray, List[str]]:
    """Average all available existing Task 3 checkpoints. Returns (probs[N,T], paths)."""
    # Find checkpoints
    candidates = []
    for fn in os.listdir("."):
        if fn.startswith("task3_checkpoint") and fn.endswith(".pt"):
            candidates.append(fn)
    if not candidates:
        raise FileNotFoundError("No existing task3 checkpoints found in cwd.")
    print(f"[T3-V2 ResNet ensemble] Using {len(candidates)} checkpoints: {candidates}")

    # Need a stable order matching test_keys.
    path_idx = {p: i for i, p in enumerate(test_keys)}
    sum_probs = np.zeros((len(test_keys), len(TAGS)), dtype=np.float64)
    count = 0
    mean, std = None, None
    for ckpt_path in candidates:
        try:
            pipeline, m, s, loaded = _load_task3_checkpoint(ckpt_path)
        except Exception as exc:
            print(f"  skip {ckpt_path}: {exc}")
            continue
        mean, std = m, s
        test_loader = _make_test_loader(
            os.path.join(DATAROOT3, "test.json"), mean, std, batch_size=12,
        )
        eval_model = pipeline.ema.ema
        eval_model.eval()
        for x, _, ps in tqdm(test_loader, desc=f"T3 RN {os.path.basename(ckpt_path)}"):
            x = x.to(pipeline.device, non_blocking=pipeline._non_blocking())
            logits = pipeline._logits_tta(x, tta_shifts, eval_model)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for j, p in enumerate(ps):
                if p in path_idx:
                    sum_probs[path_idx[p]] += probs[j]
        count += 1
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if count == 0:
        raise RuntimeError("No usable checkpoints loaded.")
    avg = sum_probs / count
    return avg.astype(np.float32), test_keys


def run3_v2(
    quick: bool = False,
    epochs: int = 12,
    ast_ckpt: str = "task3_v2_ast.pt",
    ast_weight: float = 0.70,
    resnet_weight: float = 0.30,
    skip_ast_train: bool = False,
    skip_resnet: bool = False,
    out_path: str | None = None,
) -> str:
    """Fine-tune AST, ensemble with existing ResNet34/EfficientNet checkpoints, write predictions3.json."""
    if out_path is None:
        out_path = "predictions3.json"
    print("\n=== Task 3 V2: AST fine-tune + ResNet ensemble ===")

    if skip_ast_train and os.path.isfile(ast_ckpt):
        ckpt = torch.load(ast_ckpt, map_location="cpu", weights_only=False)
        print(f"Loaded existing AST checkpoint: {ast_ckpt} (val_map={ckpt.get('val_map')})")
    else:
        ckpt = train_task3_ast(
            epochs=(2 if quick else epochs), quick=quick, save_path=ast_ckpt,
        )

    test_keys = _read_task3_test()
    ast_p, ast_paths = _predict_task3_ast_probs(ckpt, test_keys)
    # Re-order to match test_keys
    pi = {p: i for i, p in enumerate(ast_paths)}
    ast_aligned = np.zeros_like(ast_p)
    for i, k in enumerate(test_keys):
        if k in pi:
            ast_aligned[i] = ast_p[pi[k]]

    if skip_resnet:
        blend = ast_aligned
    else:
        try:
            rn_p, _ = _predict_task3_resnet_probs(test_keys, tta_shifts=(1 if quick else 5))
            w = ast_weight + resnet_weight
            blend = (ast_weight / w) * ast_aligned + (resnet_weight / w) * rn_p
            print(f"[T3-V2] blending AST(w={ast_weight}) + ResNet(w={resnet_weight})")
        except Exception as exc:
            print(f"[T3-V2] ResNet ensemble skipped: {exc}")
            blend = ast_aligned

    predictions = {
        p: {TAGS[j]: float(blend[i, j]) for j in range(len(TAGS))}
        for i, p in enumerate(test_keys)
    }
    write_submission_predictions(predictions, out_path, normalize_audio_paths=True)
    print(f"[T3-V2] wrote {out_path}")
    return out_path


# ===========================================================================
# Combined runners
# ===========================================================================
def run_all_v2(
    quick: bool = False,
    t1_epochs: int = 35,
    t2_epochs: int = 18,
    t3_epochs: int = 12,
):
    print("\n=== Pipeline: full run ===")
    t0 = time.time()
    run1_v2(quick=quick, epochs=t1_epochs)
    run2_v2(quick=quick, epochs=t2_epochs)
    run3_v2(quick=quick, epochs=t3_epochs)
    print(f"\nDone in {time.time() - t0:.1f}s")


def run_quick_v2():
    """Light end-to-end smoke test across all 3 tasks."""
    run_all_v2(quick=True)


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        run_quick_v2()
    else:
        run_all_v2()
