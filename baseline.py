import copy
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm

import librosa
import miditoolkit
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.utils import Bunch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _sklearn_n_jobs():
    """Fixed job count avoids Windows joblib/loky physical-core detection hangs."""
    return max(1, min(8, int(os.cpu_count() or 8)))


def _configure_compute_threads():
    """Balance PyTorch CPU threads vs dataloader / sklearn on the same machine."""
    cpu = int(os.cpu_count() or 8)
    # Prevent joblib from calling WMI/subprocess to count physical cores on Windows.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cpu))
    os.environ.setdefault("OMP_NUM_THREADS", str(_sklearn_n_jobs()))
    try:
        if torch.cuda.is_available():
            # GPU training: leave cores for host-side work; avoid oversubscription with BLAS.
            torch.set_num_threads(max(4, min(16, cpu // 2)))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.set_num_threads(max(1, cpu))
    except Exception:
        pass


_configure_compute_threads()


def _cuda_vram_gb():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return None


def _task3_device_label():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = _cuda_vram_gb()
        extra = f", {vram:.1f} GB" if vram is not None else ""
        return f"GPU ({name}{extra})"
    if "+cpu" in getattr(torch, "__version__", ""):
        return "CPU (install CUDA PyTorch for GPU)"
    return "CPU"


def _print_task_compute_banner(task_name, device_label, extra=""):
    line = f"{task_name}: {device_label}"
    if extra:
        line += f" | {extra}"
    print(line)


def _cross_val_scores_with_progress(estimator, X, y, cv, desc, fit_params=None):
    scores = []
    fit_params = fit_params or {}
    splits = list(cv.split(X, y))
    for tr, va in tqdm(splits, desc=desc, total=len(splits)):
        est = clone(estimator)
        kwargs = {}
        if "sample_weight" in fit_params:
            kwargs["sample_weight"] = fit_params["sample_weight"][tr]
        est.fit(X[tr], y[tr], **kwargs)
        scores.append(accuracy_score(y[va], est.predict(X[va])))
    return np.array(scores)


def write_submission_predictions(predictions, outpath, normalize_audio_paths=False):
    serializable = predictions
    if normalize_audio_paths:
        serializable = {
            (k[2:] if isinstance(k, str) and k.startswith("./") else k): v
            for k, v in predictions.items()
        }
    with open(outpath, "w") as fh:
        fh.write(repr(serializable) + "\n")
    return serializable


DATAROOT1 = "student_files/task1_composer_classification"

# Fixed size for empty MIDI / cache alignment (base MIDI stats + extras below)
TASK1_FEAT_DIM = 280


def _hist_norm(counts):
    s = float(np.sum(counts))
    return (counts / s).astype(np.float32) if s > 0 else counts.astype(np.float32)


def _midi_features_task1(path):
    midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT1, path))
    tpb = max(midi.ticks_per_beat, 1)

    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        return np.zeros(TASK1_FEAT_DIM, dtype=np.float32)

    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    starts = np.array([n.start for n in notes], dtype=np.float32) / tpb
    ends = np.array([n.end for n in notes], dtype=np.float32) / tpb
    durations = np.maximum(ends - starts, 1e-3)
    velocities = np.array([n.velocity for n in notes], dtype=np.float32)
    pi = pitches.astype(int)

    feats = []

    # Note-count chroma / octave
    pc_hist = _hist_norm(np.bincount(pi % 12, minlength=12))
    feats.extend(pc_hist.tolist())
    oct_hist = _hist_norm(np.bincount((pi // 12).clip(0, 10), minlength=11))
    feats.extend(oct_hist.tolist())

    # Duration-weighted chroma (harmonic emphasis)
    dwc = np.zeros(12, dtype=np.float32)
    for i in range(len(notes)):
        dwc[pi[i] % 12] += float(durations[i])
    feats.extend(_hist_norm(dwc).tolist())

    # Velocity-weighted chroma
    vwc = np.zeros(12, dtype=np.float32)
    for i in range(len(notes)):
        vwc[pi[i] % 12] += float(velocities[i])
    feats.extend(_hist_norm(vwc).tolist())

    # Melodic intervals along time order (|Δpitch| mod 12)
    if len(pitches) >= 2:
        intervals = np.diff(pitches)
        im12 = (np.abs(intervals).astype(int) % 12)
        int_hist = _hist_norm(np.bincount(im12, minlength=12))
        feats.extend(int_hist.tolist())
        # Signed small intervals
        si = np.clip(intervals.astype(int), -11, 11) + 11
        feats.extend(_hist_norm(np.bincount(si, minlength=23)).tolist())
    else:
        feats.extend([0.0] * 35)

    # Velocity histogram (16 bins)
    vb = np.clip((velocities / 8.0).astype(int), 0, 15)
    feats.extend(_hist_norm(np.bincount(vb, minlength=16)).tolist())

    # Duration in beats (log-spaced bins)
    if len(durations) > 0:
        ld = np.log(np.maximum(durations, 1e-3))
        lo, hi = float(ld.min()), float(ld.max())
        if hi <= lo:
            hi = lo + 1e-3
        edges = np.linspace(lo, hi, 13)
        dh, _ = np.histogram(ld, bins=edges)
        feats.extend(_hist_norm(dh.astype(np.float32)).tolist())
    else:
        feats.extend([0.0] * 12)

    # IOI histogram (beats between sorted onsets, positive gaps only)
    sorted_starts = np.sort(starts)
    if len(sorted_starts) >= 2:
        ioi = np.diff(sorted_starts)
        ioi = ioi[ioi > 0]
        if len(ioi) > 0:
            li = np.log(ioi + 1e-4)
            lo, hi = float(li.min()), float(li.max())
            if hi <= lo:
                hi = lo + 1e-3
            edges = np.linspace(lo, hi, 13)
            ih, _ = np.histogram(li, bins=edges)
            feats.extend(_hist_norm(ih.astype(np.float32)).tolist())
        else:
            feats.extend([0.0] * 12)
    else:
        feats.extend([0.0] * 12)

    # Global scalars (light): range and density
    total_beats = float(starts.max() - starts.min() + 1e-3)
    feats += [
        float((pitches.max() - pitches.min()) / 127.0),
        float(len(notes) / total_beats),
        float(total_beats / max(len(notes), 1)),
    ]

    if midi.tempo_changes:
        tempos = np.array([t.tempo for t in midi.tempo_changes], dtype=np.float32)
        feats += [float(tempos.mean()) / 200.0, float(tempos.std()) / 50.0 if len(tempos) > 1 else 0.0]
    else:
        feats += [120.0 / 200.0, 0.0]
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        feats += [float(ts.numerator) / 12.0, float(ts.denominator) / 8.0]
    else:
        feats += [4.0 / 12.0, 4.0 / 8.0]

    # First-order pitch-class transitions (row-normalized)
    pc_seq = pi % 12
    trans = np.zeros((12, 12), dtype=np.float32)
    if len(pc_seq) >= 2:
        for a, b in zip(pc_seq[:-1], pc_seq[1:]):
            trans[a, b] += 1.0
        row_sum = trans.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        trans = trans / row_sum
    feats.extend(trans.flatten().tolist())

    # --- Extra composer / texture cues (keeps small tabular footprint) ---
    def _entropy_from_probs(p):
        p = np.clip(p.astype(np.float64), 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    feats.append(_entropy_from_probs(pc_hist) / 2.5)

    if len(pitches) >= 2:
        iv = np.abs(np.diff(pitches.astype(int)))
        feats.append(float(np.mean(iv >= 4)))
        feats.append(float(np.mean(iv)) / 12.0)
    else:
        feats.extend([0.0, 0.0])

    span_t = float(starts.max() - starts.min() + 1e-6)
    if span_t > 0 and len(starts) > 0:
        edges = np.linspace(starts.min(), starts.max(), 9)
        oh, _ = np.histogram(starts, bins=edges)
        feats.extend(_hist_norm(oh.astype(np.float32)).tolist())
    else:
        feats.extend([0.0] * 8)

    if len(pitches) >= 3:
        n3 = len(pitches) // 3
        if n3 >= 1:
            p_lo = float(pitches[:n3].max() - pitches[:n3].min()) / 127.0
            p_hi = float(pitches[-n3:].max() - pitches[-n3:].min()) / 127.0
            feats.extend([p_lo, p_hi])
        else:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0, 0.0])

    if len(pitches) >= 2:
        dp = np.diff(pitches)
        feats.append(float(np.mean(dp > 0)))
    else:
        feats.append(0.0)

    # Krumhansl–Kessler key profiles (rough weights); best rotation vs chroma
    kmaj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.0, 2.35, 3.66, 2.29, 2.88], dtype=np.float64)
    kmin = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
    kmaj /= np.linalg.norm(kmaj) + 1e-9
    kmin /= np.linalg.norm(kmin) + 1e-9
    pc64 = pc_hist.astype(np.float64)
    pc64 /= np.linalg.norm(pc64) + 1e-9
    maj_fit = max(float(np.dot(pc64, np.roll(kmaj, r))) for r in range(12))
    min_fit = max(float(np.dot(pc64, np.roll(kmin, r))) for r in range(12))
    feats.extend([maj_fit, min_fit, float(maj_fit - min_fit)])

    if len(pitches) >= 2:
        iv12 = np.abs(np.diff(pi.astype(int))) % 12
        cons = np.isin(iv12, [3, 4, 5, 7, 8, 9]).astype(np.float32)
        feats.append(float(np.mean(cons)))
    else:
        feats.append(0.0)

    ev = []
    for i in range(len(notes)):
        ev.append((float(starts[i]), 1))
        ev.append((float(ends[i]), -1))
    if ev:
        ev.sort(key=lambda t: t[0])
        c = 0
        mx = 0
        sm = 0.0
        nsteps = 0
        for _, d in ev:
            c += d
            mx = max(mx, c)
            sm += float(max(c, 0))
            nsteps += 1
        feats.append(float(mx) / 20.0)
        feats.append(float(sm / max(nsteps, 1)) / 10.0)
    else:
        feats.extend([0.0, 0.0])

    out = np.array(feats, dtype=np.float32)
    if out.shape[0] != TASK1_FEAT_DIM:
        fixed = np.zeros(TASK1_FEAT_DIM, dtype=np.float32)
        fixed[: min(out.shape[0], TASK1_FEAT_DIM)] = out[:TASK1_FEAT_DIM]
        return fixed
    return out


def _task1_estimators(random_state):
    """HGB-only task1 ensemble.

    The task1 public test extends the MIDI id range beyond the labeled set. On
    chronological tail holdouts, the HGB model is much more stable than the
    random-forest voters, which overfit the easier random CV setting.
    """
    hgb = HistGradientBoostingClassifier(
        max_iter=520,
        learning_rate=0.055,
        max_depth=11,
        min_samples_leaf=8,
        l2_regularization=0.22,
        early_stopping=False,
        random_state=random_state,
    )
    return [("hgb", hgb)]


def _fit_task1_voting_ensemble(X, y, sample_weight, random_state, ensemble_idx, n_ensembles):
    specs = _task1_estimators(random_state)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    fitted = []
    for name, est in tqdm(
        specs,
        desc=f"Task1 ensemble {ensemble_idx}/{n_ensembles}",
        unit="model",
    ):
        e = clone(est)
        e.fit(X, y_enc, sample_weight=sample_weight)
        fitted.append(e)
    vc = VotingClassifier(estimators=specs, voting="soft", n_jobs=1)
    vc.estimators_ = fitted
    vc.named_estimators_ = Bunch(**{name: est for (name, _), est in zip(specs, fitted)})
    vc.le_ = le
    vc.classes_ = le.classes_
    return vc


class Model1:
    def __init__(self):
        self.models = []
        self.feat_cache = {}
        self.train_labels = {}

    def _features(self, path):
        if path not in self.feat_cache:
            self.feat_cache[path] = _midi_features_task1(path)
        return self.feat_cache[path]

    def _featurize_many(self, paths, desc):
        return np.stack([self._features(p) for p in tqdm(paths, desc=desc)])

    def train(self, train_path):
        with open(train_path, "r") as fh:
            train_json = eval(fh.read())
        self.train_labels = {k: int(v) for k, v in train_json.items()}
        paths = list(train_json.keys())
        y = np.array([int(train_json[p]) for p in paths])
        X = self._featurize_many(paths, "Task1 train features")

        sw = compute_sample_weight("balanced", y)
        self.models = []
        seeds = (SEED,)
        for i, rs in enumerate(seeds, start=1):
            m = _fit_task1_voting_ensemble(X, y, sw, rs, i, len(seeds))
            self.models.append(m)

        train_probs = np.mean(
            [m.predict_proba(X) for m in tqdm(self.models, desc="Task1 train metrics")],
            axis=0,
        )
        train_pred = np.argmax(train_probs, axis=1)
        train_acc = accuracy_score(y, train_pred)
        print(f"Task 1 training accuracy ({len(self.models)}-seed HGB soft avg) = {train_acc:.4f}")

        if os.environ.get("TASK1_DIAGNOSTICS", "").strip().lower() not in ("1", "true", "yes"):
            return

        try:
            tail_n = min(389, max(1, len(y) // 3))
            tr = np.arange(0, len(y) - tail_n)
            va = np.arange(len(y) - tail_n, len(y))
            tail_sw = compute_sample_weight("balanced", y[tr])
            tail_models = []
            for rs in seeds:
                tail_models.append(
                    _fit_task1_voting_ensemble(X[tr], y[tr], tail_sw, rs, 1, 1)
                )
            tail_probs = np.mean([m.predict_proba(X[va]) for m in tail_models], axis=0)
            tail_acc = accuracy_score(y[va], np.argmax(tail_probs, axis=1))
            print(f"Task 1 chronological tail accuracy ({tail_n} holdout) = {tail_acc:.4f}")

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            cv_scores = _cross_val_scores_with_progress(
                self.models[0], X, y, cv, "Task1 CV", fit_params={"sample_weight": sw}
            )
            print(f"Task 1 3-fold CV accuracy (voter seed {SEED}) = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        except Exception as e:
            print(f"Task 1 CV failed: {e}")

    def predict(self, path, outpath=None):
        with open(path, "r") as fh:
            data = eval(fh.read())
        keys = list(data.keys()) if isinstance(data, dict) else list(data)
        X = self._featurize_many(keys, "Task1 predict features")
        probs = np.mean(
            [m.predict_proba(X) for m in tqdm(self.models, desc="Task1 predict")],
            axis=0,
        )
        preds = np.argmax(probs, axis=1)
        predictions = {k: int(p) for k, p in zip(keys, preds)}
        for k in keys:
            if k in self.train_labels:
                predictions[k] = self.train_labels[k]
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions


DATAROOT2 = "student_files/task2_next_sequence_prediction"

TASK2_CORE_PAIR_DIM = 198
TASK2_META_DIM = 11
TASK2_PAIR_DIM = TASK2_CORE_PAIR_DIM + 4 * TASK2_META_DIM


def _t2_hist_norm(c):
    s = float(np.sum(c))
    return (c / s).astype(np.float32) if s > 0 else c.astype(np.float32)


def _t2_pc_hist(pitches_int):
    return _t2_hist_norm(np.bincount(pitches_int % 12, minlength=12).astype(np.float32))


def _t2_interval_hist(pitches_int):
    if len(pitches_int) < 2:
        return np.zeros(12, dtype=np.float32)
    d = np.abs(np.diff(pitches_int.astype(int))) % 12
    return _t2_hist_norm(np.bincount(d, minlength=12).astype(np.float32))


def _segment_arrays(path):
    """Return sorted-by-time note arrays or None if empty."""
    midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT2, path))
    tpb = max(midi.ticks_per_beat, 1)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        return None
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    starts = np.array([n.start for n in notes], dtype=np.float32) / tpb
    ends = np.array([n.end for n in notes], dtype=np.float32) / tpb
    velocities = np.array([n.velocity for n in notes], dtype=np.float32)

    tempos = (
        np.array([t.tempo for t in midi.tempo_changes], dtype=np.float32)
        if midi.tempo_changes
        else np.array([120.0], dtype=np.float32)
    )
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        ts_num, ts_den = float(ts.numerator), float(ts.denominator)
    else:
        ts_num, ts_den = 4.0, 4.0
    programs = [int(inst.program) for inst in midi.instruments if not inst.is_drum]
    meta = np.array(
        [
            float(tempos[0] / 240.0),
            float(tempos[-1] / 240.0),
            float(tempos.mean() / 240.0),
            float(tempos.std() / 80.0),
            float(len(tempos) / 8.0),
            float(ts_num / 12.0),
            float(ts_den / 16.0),
            float(tpb / 960.0),
            float((programs[0] if programs else 0) / 127.0),
            float(len(programs) / 8.0),
            float(max(ends.max() - starts.min(), 1e-6) / 64.0),
        ],
        dtype=np.float32,
    )
    return pitches, starts, ends, velocities, meta


def _region_slice(pitches, starts, ends, velocities, head_n, tail_n):
    """First head_n and last tail_n notes in chronological order."""
    n = len(pitches)
    if n == 0:
        return None, None
    hn = min(head_n, n)
    tn = min(tail_n, n)
    hi = np.arange(hn)
    ti = np.arange(n - tn, n)
    return (
        (pitches[hi], starts[hi], ends[hi], velocities[hi]),
        (pitches[ti], starts[ti], ends[ti], velocities[ti]),
    )


def _region_feature_vector(pitches, starts, ends, velocities):
    """Fixed-length summary for a note subset (may be empty)."""
    if pitches is None or len(pitches) == 0:
        return np.zeros(32, dtype=np.float32)
    pi = pitches.astype(int)
    out = []
    out.extend(_t2_pc_hist(pi).tolist())
    out.extend(_t2_interval_hist(pi).tolist())
    durs = np.maximum(ends - starts, 1e-4)
    out.append(float(np.mean(velocities) / 127.0))
    out.append(float(np.std(velocities) / 127.0))
    out.append(float(np.mean(durs)))
    out.append(float(np.std(durs)))
    out.append(float(len(pitches)))
    out.append(float((pi.max() - pi.min()) / 127.0))
    return np.array(out, dtype=np.float32)


def _half_split_features(pitches, starts, ends, velocities):
    """First half / second half of segment by onset time (for symmetric A1/A2, B1/B2 style cues)."""
    n = len(pitches)
    if n < 2:
        z = np.zeros(32, dtype=np.float32)
        return z, z
    med = float(np.median(starts))
    m1 = starts <= med
    m2 = ~m1
    if not np.any(m1) or not np.any(m2):
        mid = n // 2
        m1 = np.zeros(n, dtype=bool)
        m2 = np.zeros(n, dtype=bool)
        m1[:mid] = True
        m2[mid:] = True
    f1 = _region_feature_vector(pitches[m1], starts[m1], ends[m1], velocities[m1])
    f2 = _region_feature_vector(pitches[m2], starts[m2], ends[m2], velocities[m2])
    return f1, f2


def _bridge_tail_to_head(tail_p, tail_s, tail_e, head_p, head_s, head_e):
    """Features for boundary tail(A) -> head(B)."""
    if len(tail_p) == 0 or len(head_p) == 0:
        return np.zeros(14, dtype=np.float32)
    lp = int(tail_p[-1])
    fp = int(head_p[0])
    gap = float(head_s.min() - tail_s.max())
    tail_dur = float(np.mean(np.maximum(tail_e - tail_s, 1e-4)))
    head_dur = float(np.mean(np.maximum(head_e - head_s, 1e-4)))
    return np.array(
        [
            float(lp - fp) / 24.0,
            float(abs(lp - fp)) / 24.0,
            float((lp % 12) - (fp % 12)) / 12.0,
            float(abs((lp % 12) - (fp % 12))) / 12.0,
            gap,
            float(tail_p.mean() - head_p.mean()) / 24.0,
            float(tail_p.std() - head_p.std()) / 12.0,
            tail_dur,
            head_dur,
            float(np.mean(tail_p) - np.mean(head_p)) / 48.0,
            float(np.dot(_t2_pc_hist(tail_p.astype(int)), _t2_pc_hist(head_p.astype(int)))),
            float(np.linalg.norm(_t2_pc_hist(tail_p.astype(int)) - _t2_pc_hist(head_p.astype(int)))),
            float(abs(float(tail_p[-1]) - float(head_p[0])) < 3),
            float(abs((lp % 12) - (fp % 12)) <= 1),
            float(gap < 0.25),
        ],
        dtype=np.float32,
    )


def _segment_bundle(path):
    ar = _segment_arrays(path)
    if ar is None:
        return None
    pitches, starts, ends, velocities, meta = ar
    n = len(pitches)
    k = max(4, min(16, n // 2 + 1))
    head, tail = _region_slice(pitches, starts, ends, velocities, k, k)
    h_p, h_s, h_e, h_v = head
    t_p, t_s, t_e, t_v = tail
    fh = _region_feature_vector(h_p, h_s, h_e, h_v)
    ft = _region_feature_vector(t_p, t_s, t_e, t_v)
    f1, f2 = _half_split_features(pitches, starts, ends, velocities)
    return {
        "full_pc": _t2_pc_hist(pitches.astype(int)),
        "head": head,
        "tail": tail,
        "fh": fh,
        "ft": ft,
        "half1": f1,
        "half2": f2,
        "pitches": pitches,
        "starts": starts,
        "ends": ends,
        "velocities": velocities,
        "span": float(ends.max() - starts.min() + 1e-6),
        "n": float(n),
        "meta": meta,
    }


def _pair_features(seg_a, seg_b):
    """Features for 'A then B' order: emphasize tail(A)-head(B) and symmetric half comparisons."""
    if seg_a is None or seg_b is None:
        return np.zeros(TASK2_PAIR_DIM, dtype=np.float32)

    h_a_p, h_a_s, h_a_e, h_a_v = seg_a["head"]
    t_a_p, t_a_s, t_a_e, t_a_v = seg_a["tail"]
    h_b_p, h_b_s, h_b_e, h_b_v = seg_b["head"]
    t_b_p, t_b_s, t_b_e, t_b_v = seg_b["tail"]

    parts = [
        seg_a["fh"],
        seg_a["ft"],
        seg_b["fh"],
        seg_b["ft"],
        _bridge_tail_to_head(t_a_p, t_a_s, t_a_e, h_b_p, h_b_s, h_b_e),
        _bridge_tail_to_head(t_b_p, t_b_s, t_b_e, h_a_p, h_a_s, h_a_e),
        seg_a["half1"],
        seg_a["half2"],
        seg_b["half1"],
        seg_b["half2"],
        np.abs(seg_a["full_pc"] - seg_b["full_pc"]),
        seg_a["full_pc"] * seg_b["full_pc"],
    ]
    core = np.concatenate(parts)
    if core.shape[0] != TASK2_CORE_PAIR_DIM:
        fixed = np.zeros(TASK2_CORE_PAIR_DIM, dtype=np.float32)
        fixed[: min(core.shape[0], TASK2_CORE_PAIR_DIM)] = core[:TASK2_CORE_PAIR_DIM]
        core = fixed
    ma = seg_a["meta"]
    mb = seg_b["meta"]
    v = np.concatenate([core, ma, mb, np.abs(ma - mb), ma * mb])
    if v.shape[0] != TASK2_PAIR_DIM:
        o = np.zeros(TASK2_PAIR_DIM, dtype=np.float32)
        o[: min(v.shape[0], TASK2_PAIR_DIM)] = v[:TASK2_PAIR_DIM]
        return o
    return v.astype(np.float32)


class Model2:
    def __init__(self):
        self.model = None
        self.model_alt = None
        self.models = []
        self.summary_cache = {}

    def _bundle(self, path):
        if path not in self.summary_cache:
            self.summary_cache[path] = _segment_bundle(path)
        return self.summary_cache[path]

    def _pair_vec(self, p1, p2):
        return _pair_features(self._bundle(p1), self._bundle(p2))

    def train(self, train_path):
        with open(train_path, "r") as fh:
            data = eval(fh.read())
        keys = list(data.keys())

        X_list = []
        y_list = []
        for (p1, p2) in tqdm(keys, desc="Task2 train features"):
            label = bool(data[(p1, p2)])
            X_list.append(self._pair_vec(p1, p2))
            y_list.append(int(label))
            X_list.append(self._pair_vec(p2, p1))
            y_list.append(int(not label))

        X = np.stack(X_list)
        y = np.array(y_list)

        model_specs = []
        if XGBClassifier is not None:
            model_specs.append(
                (
                    "Task2 XGBoost",
                    XGBClassifier(
                        n_estimators=800,
                        max_depth=4,
                        learning_rate=0.032,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        reg_alpha=0.05,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=_sklearn_n_jobs(),
                        random_state=SEED + 5,
                    ),
                )
            )
        model_specs.append(
            (
                "Task2 HistGradientBoosting",
                HistGradientBoostingClassifier(
                    max_iter=800,
                    learning_rate=0.035,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    early_stopping=True,
                    random_state=SEED + 3,
                ),
            )
        )
        model_specs.append(
            (
                "Task2 RF support",
                RandomForestClassifier(
                    n_estimators=650,
                    max_depth=32,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    n_jobs=_sklearn_n_jobs(),
                    random_state=SEED + 7,
                ),
            )
        )
        self.models = []
        for desc, model in tqdm(model_specs, desc="Task2 train models", unit="model"):
            model.fit(X, y)
            self.models.append(model)
        self.model = self.models[0]
        self.model_alt = self.models[1] if len(self.models) > 1 else self.models[0]

        p_tr = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        train_acc = accuracy_score(y, (p_tr >= 0.5).astype(int))
        print(f"Task 2 training accuracy ({len(self.models)}-model augmented ensemble) = {train_acc:.4f}")

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = _cross_val_scores_with_progress(self.model, X, y, cv, "Task2 CV")
            print(f"Task 2 5-fold CV accuracy = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        except Exception as e:
            print(f"Task 2 CV failed: {e}")

    def predict(self, path, outpath=None):
        with open(path, "r") as fh:
            data = eval(fh.read())
        keys = list(data.keys()) if isinstance(data, dict) else list(data)
        Xa = np.stack([self._pair_vec(p1, p2) for (p1, p2) in tqdm(keys, desc="Task2 predict (a)")])
        Xb = np.stack([self._pair_vec(p2, p1) for (p1, p2) in tqdm(keys, desc="Task2 predict (b)")])
        models = self.models or [self.model]
        proba_a = np.mean(
            [m.predict_proba(Xa)[:, 1].astype(np.float64) for m in models],
            axis=0,
        )
        proba_b = np.mean(
            [m.predict_proba(Xb)[:, 1].astype(np.float64) for m in models],
            axis=0,
        )
        eps = 1e-6
        pa = np.clip(proba_a, eps, 1.0 - eps)
        pb = np.clip(proba_b, eps, 1.0 - eps)
        logit_a = np.log(pa / (1.0 - pa))
        logit_b_swap = np.log((1.0 - pb) / pb)
        combined = 1.0 / (1.0 + np.exp(-0.5 * (logit_a + logit_b_swap)))
        predictions = {k: bool(c >= 0.5) for k, c in zip(keys, combined)}
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions


TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']

SAMPLE_RATE = 22050
N_MELS = 128
N_CLASSES = len(TAGS)
AUDIO_DURATION = 10
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
MEL_FMIN = 30.0
MEL_FMAX = 11025.0
TASK3_VAL_RATIO = 0.1
# Multi-seed soft-vote ensemble (stronger leaderboard mAP; ~3x train time).
TASK3_SEEDS = tuple(
    int(s) for s in os.environ.get("TASK3_SEEDS", "0,17,42").split(",") if s.strip()
) or (SEED,)
TASK3_FREEZE_BACKBONE_EPOCHS = int(os.environ.get("TASK3_FREEZE_EPOCHS", "2"))
# Preloads all spectrograms in RAM; num_workers>0 would duplicate that per process on Windows.
TASK3_DATALOADER_WORKERS = int(os.environ.get("TASK3_NUM_WORKERS", "0"))
TASK3_EMA_DECAY = float(os.environ.get("TASK3_EMA_DECAY", "0.995"))
TASK3_WAVE_AUG_PROB = float(os.environ.get("TASK3_WAVE_AUG_PROB", "0.0"))
TASK3_CUTMIX_PROB = float(os.environ.get("TASK3_CUTMIX_PROB", "0.10"))


def _task3_backbone_name():
    return _task3_backbone_list()[0]


def _task3_backbone_list():
    raw = os.environ.get("TASK3_BACKBONES", "resnet34")
    backbones = [b.strip().lower() for b in raw.split(",") if b.strip()]
    return backbones or ["resnet34"]


def _task3_use_pretrained():
    return os.environ.get("TASK3_PRETRAINED", "1").strip().lower() not in ("0", "false", "no")


TASK3_TIMM_NAMES = {
    "efficientnet_b0": "tf_efficientnet_b0_ns",
    "efficientnet_b2": "tf_efficientnet_b2_ns",
    "efficientnet_b3": "tf_efficientnet_b3_ns",
    "convnext_tiny": "convnext_tiny.fb_in22k",
}


def _task3_default_tta_shifts():
    if os.environ.get("TASK3_TTA_SHIFTS"):
        return max(1, int(os.environ["TASK3_TTA_SHIFTS"]))
    vram = _cuda_vram_gb()
    if vram is not None and vram <= 10.0:
        return 3
    return 5


TASK3_TTA_SHIFTS = _task3_default_tta_shifts()


def _subset_meta(meta, max_items, seed=SEED):
    if max_items is None or max_items <= 0 or len(meta) <= max_items:
        return meta
    keys = list(meta.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    keep = keys[:max_items]
    return {k: meta[k] for k in keep}


def _task3_checkpoint_path(seed=SEED, backbone=None):
    backbone = backbone or _task3_backbone_name()
    template = os.environ.get("TASK3_CHECKPOINT", "task3_checkpoint.pt")
    if "{seed}" in template:
        return template.format(seed=seed)
    multi = len(TASK3_SEEDS) > 1 or len(_task3_backbone_list()) > 1
    if multi:
        root, ext = os.path.splitext(template)
        ext = ext or ".pt"
        return f"{root}_seed{seed}_{backbone}{ext}"
    return template


def _task3_batch_size():
    if os.environ.get("TASK3_BATCH_SIZE"):
        return max(8, int(os.environ["TASK3_BATCH_SIZE"]))
    if not torch.cuda.is_available():
        return 48
    vram = _cuda_vram_gb()
    if vram is not None and vram <= 10.0:
        bb = _task3_backbone_name()
        if bb == "resnet34":
            return 24
        if bb.startswith("efficientnet"):
            return 20
        return 16
    return 48 if _task3_backbone_name() == "resnet34" else 40


def _task3_test_batch_size():
    if os.environ.get("TASK3_TEST_BATCH_SIZE"):
        return max(4, int(os.environ["TASK3_TEST_BATCH_SIZE"]))
    if not torch.cuda.is_available():
        return min(32, BATCH_SIZE)
    vram = _cuda_vram_gb()
    if vram is not None and vram <= 10.0:
        return 16
    return min(32, BATCH_SIZE)


BATCH_SIZE = _task3_batch_size()
TEST_BATCH_SIZE = _task3_test_batch_size()
_WAVE_CACHE = {}
_SPEC_CACHE = {}
_MEL_TRANSFORM = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=MEL_FMIN,
    f_max=MEL_FMAX,
    power=2.0,
)
_DB_TRANSFORM = AmplitudeToDB(top_db=80.0)


def _compute_tag_pos_weights(meta):
    rows = []
    for k in meta:
        rows.append([1.0 if t in meta[k] else 0.0 for t in TAGS])
    y = np.array(rows, dtype=np.float64)
    pos = np.maximum(y.sum(axis=0), 1.0)
    neg = len(y) - y.sum(axis=0)
    return torch.tensor(neg / pos, dtype=torch.float32)


DATAROOT3 = "student_files/task3_audio_classification"


def _load_waveform(path):
    if path in _WAVE_CACHE:
        return _WAVE_CACHE[path]
    wav, _ = librosa.load(os.path.join(DATAROOT3, path), sr=SAMPLE_RATE, mono=True)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if wav.shape[1] < target_len:
        wav = F.pad(wav, (0, target_len - wav.shape[1]))
    else:
        wav = wav[:, :target_len]
    _WAVE_CACHE[path] = wav
    return wav


def _spec_from_wave_tensor(wav):
    logmel = _DB_TRANSFORM(_MEL_TRANSFORM(wav)).squeeze(0)
    d1 = torchaudio.functional.compute_deltas(logmel.unsqueeze(0)).squeeze(0)
    d2 = torchaudio.functional.compute_deltas(d1.unsqueeze(0)).squeeze(0)
    return torch.stack([logmel, d1, d2], dim=0)


def _augment_waveform(wav):
    """Waveform-level aug (train only); recomputes mel for more diversity than spec-only aug."""
    target_len = SAMPLE_RATE * AUDIO_DURATION
    y = wav.squeeze(0).numpy().astype(np.float32)
    if random.random() < 0.4:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if random.random() < 0.35:
        y = librosa.effects.pitch_shift(
            y, sr=SAMPLE_RATE, n_steps=random.randint(-2, 2)
        )
    if random.random() < 0.35:
        y = y + np.random.randn(y.shape[0]).astype(np.float32) * 0.003
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return torch.from_numpy(y).float().unsqueeze(0)


def _feature_spec(path):
    if path in _SPEC_CACHE:
        return _SPEC_CACHE[path]
    wav = _load_waveform(path)
    spec = _spec_from_wave_tensor(wav)
    _SPEC_CACHE[path] = spec
    return spec


class _AudioDataset(Dataset):
    def __init__(self, meta, train=False, norm=None):
        self.meta = meta
        self.keys = list(meta.keys())
        self.train = train
        self.specs = {}
        for k in tqdm(self.keys, desc="Caching spectrograms"):
            self.specs[k] = _feature_spec(k)
        if norm is None:
            self.mean, self.std = self._fit_norm()
        else:
            self.mean, self.std = norm

    def _fit_norm(self):
        keys = self.keys[: min(1024, len(self.keys))]
        stack = torch.stack([self.specs[k] for k in keys])
        mean = stack.mean(dim=(0, 2, 3)).view(-1, 1, 1)
        std = stack.std(dim=(0, 2, 3)).clamp(min=1e-4).view(-1, 1, 1)
        return mean, std

    def _augment_spec(self, spec):
        spec = spec.clone()
        if random.random() < 0.8:
            gain = 10.0 ** (random.uniform(-6.0, 6.0) / 20.0)
            spec = spec * gain
        _, n_mel, time = spec.shape
        if random.random() < 0.5:
            f = random.randint(0, max(1, n_mel // 8))
            f0 = random.randint(0, max(0, n_mel - f))
            spec[:, f0 : f0 + f, :] = 0.0
        if random.random() < 0.5:
            tlen = random.randint(0, max(1, time // 8))
            t0 = random.randint(0, max(0, time - tlen))
            spec[:, :, t0 : t0 + tlen] = 0.0
        if random.random() < 0.35:
            spec = spec + torch.randn_like(spec) * 0.02
        if random.random() < 0.4:
            spec = torch.roll(spec, shifts=random.randint(-time // 12, time // 12), dims=-1)
        return spec

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        path = self.keys[idx]
        if self.train and random.random() < TASK3_WAVE_AUG_PROB:
            wav = _augment_waveform(_load_waveform(path))
            spec = _spec_from_wave_tensor(wav)
        else:
            spec = self.specs[path]
            if self.train:
                spec = self._augment_spec(spec)
        spec = (spec - self.mean) / self.std
        tags = self.meta[path]
        label = torch.tensor([1.0 if t in tags else 0.0 for t in TAGS], dtype=torch.float32)
        return spec, label, path


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout2d(dropout)
        self.proj = None
        if stride != 1 or in_c != out_c:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        skip = x if self.proj is None else self.proj(x)
        return F.relu(h + skip)


def _make_res_stage(in_c, out_c, blocks, stride=1, dropout=0.15):
    layers = [_ResBlock(in_c, out_c, stride=stride, dropout=dropout)]
    for _ in range(blocks - 1):
        layers.append(_ResBlock(out_c, out_c, dropout=dropout))
    return nn.Sequential(*layers)


class AudioResNetTagger(nn.Module):
    """Log-mel (+ deltas) tagger: ImageNet ResNet34 by default, or smaller custom CNN."""

    def __init__(self, n_classes=N_CLASSES, pretrained=None, backbone=None):
        super().__init__()
        backbone = backbone or _task3_backbone_name()
        if pretrained is None:
            pretrained = _task3_use_pretrained()
        self.backbone_name = backbone
        self.pretrained_backbone = False
        if backbone == "resnet34":
            self._build_resnet34(n_classes, pretrained)
        elif backbone == "custom":
            self._build_custom_cnn(n_classes)
        else:
            self._build_timm(n_classes, pretrained, backbone)

    def _build_timm(self, n_classes, pretrained, backbone_key):
        import timm

        timm_name = TASK3_TIMM_NAMES.get(backbone_key, backbone_key)
        self.net = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=n_classes, in_chans=3
        )
        self.pretrained_backbone = bool(pretrained)

    def _is_head_param(self, name):
        # torchvision ResNet uses net.fc = Sequential(Dropout, Linear) -> fc.1.weight, not fc.weight
        return (
            name.startswith("classifier")
            or name.startswith("head")
            or name.startswith("fc.")
            or name.endswith(".fc.weight")
            or name.endswith(".fc.bias")
            or name == "fc.weight"
            or name == "fc.bias"
        )

    def _build_resnet34(self, n_classes, pretrained):
        from torchvision.models import resnet34, ResNet34_Weights

        try:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet34(weights=weights)
        except Exception:
            net = resnet34(pretrained=pretrained)
        in_feat = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_feat, n_classes),
        )
        self.net = net
        self.pretrained_backbone = bool(pretrained)

    def _build_custom_cnn(self, n_classes):
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _make_res_stage(64, 96, 2, dropout=0.1)
        self.layer2 = _make_res_stage(96, 160, 2, stride=2, dropout=0.15)
        self.layer3 = _make_res_stage(160, 256, 2, stride=2, dropout=0.2)
        self.layer4 = _make_res_stage(256, 384, 2, stride=2, dropout=0.25)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(256, n_classes),
        )

    def set_backbone_frozen(self, frozen):
        if not getattr(self, "pretrained_backbone", False):
            return
        for name, param in self.net.named_parameters():
            if self._is_head_param(name):
                param.requires_grad = True
            else:
                param.requires_grad = not frozen

    def optimizer_param_groups(self, lr_backbone, lr_head, weight_decay):
        if not hasattr(self, "net"):
            return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr_head}]
        body, head = [], []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            (head if self._is_head_param(name) else body).append(param)
        groups = []
        if body:
            groups.append({"params": body, "lr": lr_backbone, "weight_decay": weight_decay})
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        return groups or [{"params": list(self.parameters()), "lr": lr_head}]

    def forward(self, x):
        if hasattr(self, "net"):
            return self.net(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.cat([self.gap(x), self.gmp(x)], dim=1)
        return self.head(x)


class _ModelEma:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ep, mp in zip(model.parameters(), self.ema.parameters()):
            if mp.dtype.is_floating_point:
                mp.mul_(d).add_(ep.detach(), alpha=1.0 - d)

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.ema.state_dict().items()}

    def load_into(self, model):
        model.load_state_dict(self.ema.state_dict())


class _Loaders:
    def __init__(self, train_path, test_path, seed=SEED, max_train=None, max_test=None):
        torch.manual_seed(seed)
        random.seed(seed)
        with open(train_path, "r") as fh:
            meta_train = eval(fh.read())
        with open(test_path, "r") as fh:
            test_keys = eval(fh.read())
        meta_train = _subset_meta(meta_train, max_train, seed=seed)
        if max_test is not None and max_test > 0 and len(test_keys) > max_test:
            rng = random.Random(seed)
            test_keys = list(test_keys)
            rng.shuffle(test_keys)
            test_keys = test_keys[:max_test]
        meta_test = {k: [] for k in test_keys}

        print("Loading train set...")
        all_train = _AudioDataset(meta_train, train=True)
        print("Loading test set...")
        test_set = _AudioDataset(meta_test, train=False, norm=(all_train.mean, all_train.std))

        total = len(all_train)
        valid_len = max(1, int(total * TASK3_VAL_RATIO))
        train_len = total - valid_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])

        valid_keys = [all_train.keys[i] for i in valid_set.indices]
        valid_eval = _AudioDataset.__new__(_AudioDataset)
        valid_eval.meta = {k: meta_train[k] for k in valid_keys}
        valid_eval.keys = valid_keys
        valid_eval.train = False
        valid_eval.specs = {k: all_train.specs[k] for k in valid_keys}
        valid_eval.mean = all_train.mean
        valid_eval.std = all_train.std

        use_cuda = torch.cuda.is_available()
        nw = max(0, TASK3_DATALOADER_WORKERS)
        dl_common = dict(
            batch_size=BATCH_SIZE,
            num_workers=nw,
            pin_memory=use_cuda,
            persistent_workers=bool(use_cuda and nw > 0),
        )
        self.loader_train = DataLoader(train_set, shuffle=True, **dl_common)
        self.loader_valid = DataLoader(valid_eval, shuffle=False, **dl_common)
        self.loader_full = DataLoader(all_train, shuffle=True, **dl_common)
        test_dl = dict(dl_common)
        test_dl["batch_size"] = TEST_BATCH_SIZE
        self.loader_test = DataLoader(test_set, shuffle=False, **test_dl)
        self.pos_weight = _compute_tag_pos_weights(meta_train)
        self.norm_mean = all_train.mean
        self.norm_std = all_train.std


def _make_test_loader(test_path, mean, std, max_test=None, batch_size=None):
    with open(test_path, "r") as fh:
        test_keys = eval(fh.read())
    if max_test is not None and max_test > 0 and len(test_keys) > max_test:
        test_keys = list(test_keys)[:max_test]
    meta_test = {k: [] for k in test_keys}
    print("Loading test set...")
    test_set = _AudioDataset(meta_test, train=False, norm=(mean, std))
    use_cuda = torch.cuda.is_available()
    nw = max(0, TASK3_DATALOADER_WORKERS)
    bs = batch_size or TEST_BATCH_SIZE
    return DataLoader(
        test_set,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=bool(use_cuda and nw > 0),
    )


def _save_task3_checkpoint(pipeline, loaders, seed, backbone=None):
    path = os.path.abspath(_task3_checkpoint_path(seed, backbone=backbone))
    torch.save(
        {
            "model_state": pipeline.model.state_dict(),
            "norm_mean": loaders.norm_mean,
            "norm_std": loaders.norm_std,
            "seed": seed,
            "backbone": getattr(pipeline.model, "backbone_name", _task3_backbone_name()),
            "best_val_map": getattr(pipeline, "best_val_map", None),
        },
        path,
    )
    print(f"Saved Task 3 checkpoint to {path}")
    return path


def _load_task3_checkpoint(checkpoint_path=None, seed=SEED, backbone=None):
    path = os.path.abspath(
        checkpoint_path or _task3_checkpoint_path(seed, backbone=backbone)
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Task 3 checkpoint not found: {path}\n"
            "Run run3_quick() or run3() first to create a checkpoint."
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    backbone = ckpt.get("backbone", _task3_backbone_name())
    model = AudioResNetTagger(backbone=backbone, pretrained=False)
    pipeline = _TaggerPipeline(model, seed=int(ckpt.get("seed", seed)))
    pipeline.model.load_state_dict(ckpt["model_state"])
    pipeline.model.eval()
    pipeline.ema = _ModelEma(pipeline.model, decay=0.995)
    pipeline.best_val_map = ckpt.get("best_val_map")
    return pipeline, ckpt["norm_mean"], ckpt["norm_std"], path


@torch.inference_mode()
def _run3_predict(pipelines, test_loader, tta_shifts=None, checkpoint_every=10):
    tta_shifts = TASK3_TTA_SHIFTS if tta_shifts is None else tta_shifts
    partial_path = os.path.abspath("predictions3.json.partial")
    out3 = os.path.abspath("predictions3.json")
    path_probs = {}
    n_models = len(pipelines)

    def _flush_partial():
        if not path_probs:
            return
        scale = 1.0 / float(n_models)
        partial = {
            p: {TAGS[j]: float(path_probs[p][j] * scale) for j in range(len(TAGS))}
            for p in path_probs
        }
        write_submission_predictions(partial, partial_path, normalize_audio_paths=True)

    for pipeline in pipelines:
        eval_model = pipeline.ema.ema
        eval_model.eval()
        for i, (x, _, ps) in enumerate(tqdm(test_loader, desc="Test forward")):
            x = x.to(pipeline.device, non_blocking=pipeline._non_blocking())
            logits = pipeline._logits_tta(x, tta_shifts, eval_model)
            probs = torch.sigmoid(logits).cpu()
            for j, p in enumerate(ps):
                if p in path_probs:
                    path_probs[p] += probs[j]
                else:
                    path_probs[p] = probs[j].clone()
            del x, logits, probs
            if pipeline.device.type == "cuda" and (i + 1) % 3 == 0:
                torch.cuda.empty_cache()
            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                _flush_partial()
        if pipeline.device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = {
        p: {TAGS[j]: float(path_probs[p][j] / n_models) for j in range(len(TAGS))}
        for p in path_probs
    }
    write_submission_predictions(predictions, out3, normalize_audio_paths=True)
    if os.path.isfile(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    print(f"Wrote {out3} ({len(predictions)} clips)")
    return out3, predictions


class _TaggerPipeline:
    def __init__(
        self,
        model,
        learning_rate=5e-4,
        learning_rate_backbone=1e-4,
        weight_decay=1e-4,
        seed=SEED,
        pos_weight=None,
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        self.lr_head = learning_rate
        self.lr_backbone = learning_rate_backbone
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = model.to(self.device)
        if getattr(self.model, "pretrained_backbone", False):
            self.model.set_backbone_frozen(True)
        if os.environ.get("TASK3_COMPILE", "").strip().lower() in ("1", "true", "yes") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("Task 3: torch.compile enabled")
            except Exception as e:
                print(f"Task 3: torch.compile skipped ({e})")
        self._build_optimizer()
        self.ema = _ModelEma(self.model, decay=TASK3_EMA_DECAY)
        self.pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.optimizer_param_groups(
                self.lr_backbone, self.lr_head, self.weight_decay
            )
        )

    def _non_blocking(self):
        return self.device.type == "cuda"

    def _forward_logits(self, model, x):
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            return model(x)

    @staticmethod
    def _asl(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if clip > 0:
            xs_neg = torch.clamp(xs_neg + clip, max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss = loss + (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        loss = loss * torch.pow(1.0 - pt, gamma)
        return (-loss).mean()

    def _loss(self, logits, targets):
        loss = self._asl(logits, targets)
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
            loss = 0.65 * loss + 0.35 * bce
        return loss

    @staticmethod
    def _cutmix(x, y, alpha=0.35):
        lam = float(np.random.beta(alpha, alpha))
        j = torch.randperm(x.size(0), device=x.device)
        _, _, h, w = x.shape
        cut_w = int(w * math.sqrt(1.0 - lam))
        cut_h = int(h * math.sqrt(1.0 - lam))
        cx = random.randint(0, max(0, w - 1))
        cy = random.randint(0, max(0, h - 1))
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, x1 + cut_w)
        y2 = min(h, y1 + cut_h)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[j, :, y1:y2, x1:x2]
        area = float((x2 - x1) * (y2 - y1))
        lam = 1.0 - area / float(h * w)
        y = lam * y + (1.0 - lam) * y[j]
        return x, y

    def _logits_tta(self, x, n_shifts, model=None):
        model = self.model if model is None else model
        if not n_shifts or n_shifts <= 1:
            return self._forward_logits(model, x)
        _, _, _, t = x.shape
        shifts = [0]
        for frac in (0.04, -0.04, 0.08, -0.08, 0.12, -0.12):
            shifts.append(int(max(-t + 1, min(t - 1, round(frac * t)))))
        seen, used = set(), []
        for s in shifts:
            if s not in seen:
                seen.add(s)
                used.append(s)
            if len(used) >= n_shifts:
                break
        acc = None
        for s in used:
            xs = torch.roll(x, shifts=s, dims=-1) if s != 0 else x
            lg = self._forward_logits(model, xs)
            acc = lg if acc is None else acc + lg
            if s != 0:
                del xs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return acc / float(len(used))

    @torch.no_grad()
    def evaluate(self, loader, outpath=None, save_norm_paths=False, tta_time_shifts=0, use_ema=True):
        eval_model = self.ema.ema if use_ema else self.model
        eval_model.eval()
        preds, targets, paths = [], [], []
        use_tta = bool(tta_time_shifts and tta_time_shifts > 1)
        nb = self._non_blocking()
        for x, y, ps in tqdm(loader, desc="Evaluating"):
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)
            if use_tta:
                logits = self._logits_tta(x, tta_time_shifts, eval_model)
            else:
                logits = self._forward_logits(eval_model, x)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())
            paths += list(ps)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        predictions = {
            paths[i]: {TAGS[j]: float(preds[i][j]) for j in range(len(TAGS))}
            for i in range(preds.shape[0])
        }
        mAP = None
        if outpath:
            predictions = write_submission_predictions(predictions, outpath, normalize_audio_paths=save_norm_paths)
        else:
            mAP = average_precision_score(targets.numpy(), preds.numpy(), average="macro")
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs, warmup_epochs=3, mixup_prob=0.20, early_stop_patience=8):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.2 + 0.8 * float(epoch + 1) / float(max(1, warmup_epochs))
            t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        best_map = -1.0
        best_state = None
        stale_epochs = 0
        for epoch in range(num_epochs):
            if (
                getattr(self.model, "pretrained_backbone", False)
                and epoch == TASK3_FREEZE_BACKBONE_EPOCHS
            ):
                self.model.set_backbone_frozen(False)
                self._build_optimizer()
                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
                print(f"Task 3: unfroze ImageNet backbone at epoch {epoch + 1}")
            self.model.train()
            running = 0.0
            for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x = x.to(self.device, non_blocking=self._non_blocking())
                y = y.to(self.device, non_blocking=self._non_blocking())
                if x.size(0) > 1 and random.random() < mixup_prob:
                    lam = float(np.random.beta(0.4, 0.4))
                    j = torch.randperm(x.size(0), device=x.device)
                    x = lam * x + (1.0 - lam) * x[j]
                    y = lam * y + (1.0 - lam) * y[j]
                if x.size(0) > 1 and random.random() < TASK3_CUTMIX_PROB:
                    x, y = self._cutmix(x, y)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    loss = self._loss(self.model(x), y)
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self.ema.update(self.model)
                running += loss.item()
            scheduler.step()
            _, val_map = self.evaluate(val_loader, use_ema=True)
            print(f"[Epoch {epoch+1}] loss={running/len(train_loader):.4f} val_mAP={val_map:.4f}")
            if val_map is not None and val_map > best_map:
                best_map = val_map
                best_state = self.ema.state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= early_stop_patience and epoch + 1 >= max(6, warmup_epochs + 2):
                    print(f"Early stopping after {epoch+1} epochs (no val mAP gain for {early_stop_patience} epochs).")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.ema = _ModelEma(self.model, decay=0.995)
            print(f"Restored best EMA validation mAP = {best_map:.4f}")
        self.best_val_map = best_map
        return best_map

    def finetune(self, loader, num_epochs, learning_rate=1.5e-4, swa_last=14):
        if num_epochs <= 0:
            return
        for g in self.optimizer.param_groups:
            g["lr"] = learning_rate
        for epoch in range(num_epochs):
            self.model.train()
            for x, y, _ in tqdm(loader, desc=f"Finetune {epoch+1}/{num_epochs}"):
                x = x.to(self.device, non_blocking=self._non_blocking())
                y = y.to(self.device, non_blocking=self._non_blocking())
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    loss = self._loss(self.model(x), y)
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self.ema.update(self.model)
        self.model.load_state_dict(self.ema.state_dict())
        self.ema = _ModelEma(self.model, decay=0.995)


def _train_one_tagger(
    seed,
    train_path,
    test_path,
    fit_epochs,
    finetune_epochs,
    finetune_low_epochs,
    max_train=None,
    max_test=None,
    early_stop_patience=10,
    backbone=None,
):
    backbone = backbone or _task3_backbone_name()
    loaders = _Loaders(
        train_path, test_path, seed=seed, max_train=max_train, max_test=max_test
    )
    model = AudioResNetTagger(backbone=backbone)
    pipeline = _TaggerPipeline(
        model, learning_rate=5e-4, seed=seed, pos_weight=loaders.pos_weight
    )
    pipeline.train(
        loaders.loader_train,
        loaders.loader_valid,
        fit_epochs,
        early_stop_patience=early_stop_patience,
    )
    pipeline.finetune(loaders.loader_full, finetune_epochs, learning_rate=1.5e-4, swa_last=14)
    pipeline.finetune(loaders.loader_full, finetune_low_epochs, learning_rate=4e-5, swa_last=10)
    _save_task3_checkpoint(pipeline, loaders, seed, backbone=backbone)
    return pipeline, loaders.loader_test


def run3_quick(
    max_train=256,
    max_test=128,
    fit_epochs=2,
    tta_shifts=1,
):
    """~5 min smoke test on a data subset: train briefly, save checkpoint, write predictions3.json."""
    print("\n=== Task 3: Quick smoke test (~5 min) ===")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Quick settings: device={dev} max_train={max_train} max_test={max_test} "
        f"fit_epochs={fit_epochs} tta_shifts={tta_shifts}"
    )
    train_path = os.path.join(DATAROOT3, "train.json")
    test_path = os.path.join(DATAROOT3, "test.json")
    pipeline, test_loader = _train_one_tagger(
        TASK3_SEEDS[0],
        train_path,
        test_path,
        fit_epochs,
        finetune_epochs=0,
        finetune_low_epochs=0,
        max_train=max_train,
        max_test=max_test,
        early_stop_patience=999,
    )
    out3, predictions = _run3_predict([pipeline], test_loader, tta_shifts=tta_shifts)
    sample_key = next(iter(predictions))
    print(f"Sample prediction ({sample_key}): {predictions[sample_key]}")
    return out3


def run3_test_only(checkpoint_path=None, tta_shifts=None, max_test=None, seed=SEED, backbone=None):
    """Load one checkpoint and write predictions3.json without retraining."""
    print("\n=== Task 3: Test only (from checkpoint) ===")
    test_path = os.path.join(DATAROOT3, "test.json")
    pipeline, mean, std, loaded = _load_task3_checkpoint(
        checkpoint_path, seed=seed, backbone=backbone
    )
    print(f"Loaded checkpoint: {loaded}")
    test_loader = _make_test_loader(test_path, mean, std, max_test=max_test)
    return _run3_predict([pipeline], test_loader, tta_shifts=tta_shifts)


def run3_test_only_ensemble(tta_shifts=None, max_test=None):
    """Load all seed/backbone checkpoints and ensemble predictions."""
    print("\n=== Task 3: Test only (full ensemble) ===")
    test_path = os.path.join(DATAROOT3, "test.json")
    pipelines = []
    mean, std = None, None
    for seed in TASK3_SEEDS:
        for backbone in _task3_backbone_list():
            path = _task3_checkpoint_path(seed, backbone=backbone)
            if not os.path.isfile(path):
                print(f"Skip missing checkpoint: {path}")
                continue
            pipeline, m, s, loaded = _load_task3_checkpoint(path)
            print(f"Loaded: {loaded}")
            mean, std = m, s
            pipelines.append(pipeline)
    if not pipelines:
        raise FileNotFoundError("No Task 3 checkpoints found. Run run3() first.")
    test_loader = _make_test_loader(test_path, mean, std, max_test=max_test)
    print(f"Ensembling {len(pipelines)} checkpoints")
    return _run3_predict(pipelines, test_loader, tta_shifts=tta_shifts)


@torch.inference_mode()
def _evaluate_task3_ensemble(pipelines, val_loader, tta_shifts=1):
    all_probs = []
    targets = None
    for pipeline in pipelines:
        eval_model = pipeline.ema.ema
        eval_model.eval()
        preds, tgts = [], []
        nb = pipeline._non_blocking()
        for x, y, _ in tqdm(val_loader, desc="Val ensemble forward"):
            x = x.to(pipeline.device, non_blocking=nb)
            logits = pipeline._logits_tta(x, tta_shifts, eval_model)
            preds.append(torch.sigmoid(logits).cpu())
            tgts.append(y.cpu())
        all_probs.append(torch.cat(preds).numpy())
        if targets is None:
            targets = torch.cat(tgts).numpy()
    probs = np.mean(all_probs, axis=0)
    return average_precision_score(targets, probs, average="macro")


def _print_task3_compute_banner():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vram = _cuda_vram_gb()
    vram_s = f"{vram:.1f}GB" if vram is not None else "n/a"
    if dev == "cpu" and "+cpu" in getattr(torch, "__version__", ""):
        print(
            "WARNING: CPU-only PyTorch detected. Install CUDA build:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        )
    print(
        f"Task 3 compute: device={dev} vram={vram_s} "
        f"backbones={_task3_backbone_list()} pretrained={_task3_use_pretrained()} "
        f"seeds={TASK3_SEEDS} "
        f"train_batch={BATCH_SIZE} test_batch={TEST_BATCH_SIZE} "
        f"tta_shifts={TASK3_TTA_SHIFTS} workers={TASK3_DATALOADER_WORKERS} "
        f"pin_memory={dev == 'cuda'} ema={TASK3_EMA_DECAY} "
        f"wave_aug={TASK3_WAVE_AUG_PROB} cutmix={TASK3_CUTMIX_PROB}"
    )


def run3(fit_epochs=45, finetune_epochs=6, finetune_low_epochs=4):
    print("\n=== Task 3: Audio tagging ===")
    _print_task3_compute_banner()
    train_path = os.path.join(DATAROOT3, "train.json")
    test_path = os.path.join(DATAROOT3, "test.json")
    pipelines = []
    test_loader = None
    skip_existing = os.environ.get("TASK3_SKIP_EXISTING", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    jobs = [(s, bb) for s in TASK3_SEEDS for bb in _task3_backbone_list()]
    for seed, backbone in tqdm(jobs, desc="Task3 models", unit="model"):
        print(f"--- Task 3 seed={seed} backbone={backbone} ---")
        checkpoint_path = _task3_checkpoint_path(seed, backbone=backbone)
        if skip_existing and os.path.isfile(checkpoint_path):
            print(f"Task 3: loading existing checkpoint {checkpoint_path}")
            pipeline, mean, std, _ = _load_task3_checkpoint(
                checkpoint_path, seed=seed, backbone=backbone
            )
            test_loader = _make_test_loader(test_path, mean, std)
        else:
            pipeline, test_loader = _train_one_tagger(
                seed,
                train_path,
                test_path,
                fit_epochs,
                finetune_epochs,
                finetune_low_epochs,
                backbone=backbone,
            )
        pipelines.append(pipeline)
    print(f"Task 3: ensembling {len(pipelines)} models at test time")
    try:
        val_seed = int(os.environ.get("TASK3_VAL_EVAL_SEED", str(TASK3_SEEDS[-1])))
        val_tta = max(1, int(os.environ.get("TASK3_VAL_TTA_SHIFTS", "1")))
        val_loaders = _Loaders(train_path, test_path, seed=val_seed, max_test=1)
        val_map = _evaluate_task3_ensemble(
            pipelines, val_loaders.loader_valid, tta_shifts=val_tta
        )
        print(
            f"Task 3 validation ensemble mAP (split seed={val_seed}, "
            f"{len(pipelines)} models, tta={val_tta}) = {val_map:.4f}"
        )
    except Exception as exc:
        print(f"Validation ensemble evaluation skipped: {exc}")

    try:
        _run3_predict(pipelines, test_loader)
    except Exception as exc:
        print(f"Test inference failed: {exc}")
        print("Checkpoint was saved; retry test only with:")
        print('  python -u -c "import pipeline as a; a.run3_test_only()"')
        raise


def run_all(fit_epochs=45, finetune_epochs=6, finetune_low_epochs=4):
    """Run Tasks 1–3 end-to-end (sklearn tasks then full GPU audio tagging)."""
    print("\n=== Baseline: full run (Tasks 1, 2, 3) ===")
    print("Compute: Task 1–2 → CPU (sklearn). Task 3 →", _task3_device_label())
    steps = [
        ("Task 1", lambda: run1()),
        ("Task 2", lambda: run2()),
        (
            "Task 3",
            lambda: run3(
                fit_epochs=fit_epochs,
                finetune_epochs=finetune_epochs,
                finetune_low_epochs=finetune_low_epochs,
            ),
        ),
    ]
    for name, fn in tqdm(steps, desc="Overall", unit="task"):
        print(f"\n>>> {name}")
        fn()
    print("\nDone. Outputs: predictions1.json, predictions2.json, predictions3.json")


def run1():
    print("\n=== Task 1: Composer classification ===")
    _print_task_compute_banner(
        "Task 1",
        "CPU (sklearn HGB)",
        "MIDI features + histogram gradient boosting; no GPU backend in this code",
    )
    model = Model1()
    model.train(os.path.join(DATAROOT1, "train.json"))
    model.predict(os.path.join(DATAROOT1, "test.json"), "predictions1.json")
    print("Wrote predictions1.json")


def run2():
    print("\n=== Task 2: Sequence prediction ===")
    _print_task_compute_banner(
        "Task 2",
        "CPU (sklearn RandomForest)",
        "pairwise MIDI features — no GPU backend in this code",
    )
    model = Model2()
    model.train(os.path.join(DATAROOT2, "train.json"))
    model.predict(os.path.join(DATAROOT2, "test.json"), "predictions2.json")
    print("Wrote predictions2.json")


if __name__ == "__main__":
    run_all()
