import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import ndimage, signal, spatial


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def infer_stride(coords_yx) -> int:
    coords = _to_numpy(coords_yx).astype(np.int64).reshape(-1, 2)
    strides = []
    for axis in range(2):
        vals = np.unique(coords[:, axis])
        if len(vals) < 2:
            continue
        diffs = np.abs(np.diff(vals))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            continue
        strides.append(int(np.gcd.reduce(diffs)))
    return min(strides) if strides else 1


def coords_to_grid(coords_yx, stride=None):
    coords = _to_numpy(coords_yx).astype(np.int64).reshape(-1, 2)
    if stride is None:
        stride = infer_stride(coords)
    stride = max(int(stride), 1)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64), stride
    grid = (coords - coords.min(axis=0)) // stride
    return grid.astype(np.int64), stride


def _raster(grid, shape=None):
    if shape is None:
        shape = tuple((grid.max(axis=0) + 1).tolist()) if len(grid) else (0, 0)
    occ = np.zeros(shape, dtype=bool)
    if len(grid):
        occ[grid[:, 0], grid[:, 1]] = True
    return occ


def tissue_components(grid) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64).reshape(-1, 2)
    if len(grid) == 0:
        return np.zeros(0, dtype=np.int64)
    grid = grid - grid.min(axis=0)
    labels, _ = ndimage.label(_raster(grid), structure=np.ones((3, 3), dtype=bool))
    return (labels[grid[:, 0], grid[:, 1]] - 1).astype(np.int64)


def convex_hull_mask(cells, shape) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    mask = np.zeros(shape, dtype=bool)
    if len(cells) == 0:
        return mask
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < shape[1])
    mask[cells[inside, 0], cells[inside, 1]] = True

    pts = np.unique(cells, axis=0).astype(np.float64)
    if len(pts) < 3 or np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return mask
    try:
        hull = spatial.ConvexHull(pts)
    except Exception:
        return mask

    lo = np.maximum(cells.min(axis=0), 0)
    hi = np.minimum(cells.max(axis=0) + 1, np.asarray(shape))
    if (hi <= lo).any():
        return mask
    rr, cc = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
    cand = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)
    eq = hull.equations
    ok = (cand @ eq[:, :2].T + eq[:, 2] <= 1e-9).all(axis=1)
    sel = cand[ok].astype(np.int64)
    mask[sel[:, 0], sel[:, 1]] = True
    return mask


def _clear_region(available, mask, ty, tx):
    H, W = available.shape
    h, w = mask.shape
    y0, x0 = max(ty, 0), max(tx, 0)
    y1, x1 = min(ty + h, H), min(tx + w, W)
    if y1 <= y0 or x1 <= x0:
        return
    available[y0:y1, x0:x1] &= ~mask[y0 - ty : y1 - ty, x0 - tx : x1 - tx]


def align_moving_to_reference(grid_r, grid_m, comp_m=None) -> np.ndarray:
    grid_r = np.asarray(grid_r, dtype=np.int64).reshape(-1, 2)
    grid_m = np.asarray(grid_m, dtype=np.int64).reshape(-1, 2)
    aligned = grid_m.copy()
    if len(grid_m) == 0 or len(grid_r) == 0:
        return aligned
    if comp_m is None:
        comp_m = tissue_components(grid_m)
    comp_m = np.asarray(comp_m, dtype=np.int64).reshape(-1)

    shape_r = tuple((grid_r.max(axis=0) + 1).tolist())
    comp_r = tissue_components(grid_r)
    available = np.zeros(shape_r, dtype=bool)
    for cid in np.unique(comp_r):
        available |= convex_hull_mask(grid_r[comp_r == cid], shape_r)
    ref_centroid = grid_r.mean(axis=0)

    ids, counts = np.unique(comp_m, return_counts=True)
    order = ids[np.lexsort((ids, -counts))]
    for cid in order:
        idx = np.nonzero(comp_m == cid)[0]
        cells = grid_m[idx]
        origin = cells.min(axis=0)
        local = cells - origin
        h, w = (local.max(axis=0) + 1).tolist()
        hull = convex_hull_mask(local, (h, w))

        overlap = np.rint(
            signal.fftconvolve(available.astype(np.float32), hull[::-1, ::-1].astype(np.float32), mode="full")
        )
        best = overlap.max() if overlap.size else 0.0
        if best >= 1.0:
            ks = np.argwhere(overlap == best)
            t = ks - np.array([h - 1, w - 1])
            theta = t - origin
            pick = int(np.argmin((theta**2).sum(axis=1)))
            theta = theta[pick]
        else:
            theta = np.rint(ref_centroid - cells.mean(axis=0)).astype(np.int64)

        ty, tx = (origin + theta).tolist()
        _clear_region(available, hull, ty, tx)
        aligned[idx] = cells + theta
    return aligned


@dataclass
class PCAStats:
    mean: torch.Tensor
    components: torch.Tensor
    d_min: float
    d_max: float

    def project(self, x) -> torch.Tensor:
        x = torch.as_tensor(x).float()
        return (x - self.mean) @ self.components

    def to_dict(self):
        return {
            "mean": self.mean.float().cpu(),
            "components": self.components.float().cpu(),
            "d_min": float(self.d_min),
            "d_max": float(self.d_max),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            mean=torch.as_tensor(d["mean"]).float(),
            components=torch.as_tensor(d["components"]).float(),
            d_min=float(d["d_min"]),
            d_max=float(d["d_max"]),
        )

    def save(self, path):
        torch.save(self.to_dict(), str(path))

    @classmethod
    def load(cls, path):
        return cls.from_dict(torch.load(str(path), map_location="cpu", weights_only=True))


def fit_pca_stats(feature_iter, k=16, max_tiles=200_000, num_pairs=100_000, seed=0, per_slide=512) -> PCAStats:
    rng = np.random.default_rng(seed)
    chunks, slide_ids, total = [], [], 0
    for sid, feats in enumerate(feature_iter):
        if total >= max_tiles:
            break
        feats = torch.as_tensor(feats).float()
        feats = feats.reshape(-1, feats.shape[-1])
        n = feats.shape[0]
        if n == 0:
            continue
        take = min(n, per_slide, max_tiles - total)
        sel = np.sort(rng.choice(n, size=take, replace=False))
        chunks.append(feats[torch.as_tensor(sel)])
        slide_ids.append(np.full(take, sid, dtype=np.int64))
        total += take
    if not chunks:
        raise ValueError("fit_pca_stats: no features provided")
    x = torch.cat(chunks, dim=0)
    sids = np.concatenate(slide_ids)
    M, D = x.shape

    mean = x.mean(dim=0)
    centered = x - mean
    q = max(1, min(int(k), D, M))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, V = torch.pca_lowrank(centered, q=q, center=False)
    components = V[:, :q].contiguous().float()
    proj = centered @ components

    i = rng.integers(0, M, size=num_pairs)
    j = rng.integers(0, M, size=num_pairs)
    if len(np.unique(sids)) > 1:
        same = sids[i] == sids[j]
        for _ in range(100):
            if not same.any():
                break
            j[same] = rng.integers(0, M, size=int(same.sum()))
            same = sids[i] == sids[j]
        i, j = i[~same], j[~same]
    if len(i) == 0:
        d_min, d_max = 0.0, 1.0
    else:
        d = ((proj[torch.as_tensor(i)] - proj[torch.as_tensor(j)]) ** 2).sum(dim=1)
        d_min, d_max = float(d.min()), float(d.max())
    if d_max <= d_min:
        d_max = d_min + 1e-6
    return PCAStats(mean=mean.float(), components=components, d_min=d_min, d_max=d_max)


def pca_distance_norm(stats, a, b) -> torch.Tensor:
    d = ((stats.project(a) - stats.project(b)) ** 2).sum(dim=1)
    return ((d - stats.d_min) / (stats.d_max - stats.d_min)).clamp(0.0, 1.0)


@dataclass
class MixResult:
    features: torch.Tensor
    source: torch.Tensor
    moving_index: torch.Tensor
    n_eligible: int
    n_swapped: int


def _roi_array(roi, n):
    if roi is None:
        return np.ones(n, dtype=bool)
    roi = _to_numpy(roi).astype(bool).reshape(-1)
    if len(roi) != n:
        raise ValueError(f"ROI mask length {len(roi)} != number of tiles {n}")
    return roi


def its_mix(
    feat_r, coords_r, feat_m, coords_m, N, P, T_PCA, roi_r=None, roi_m=None, pca_stats=None, rng=None
) -> MixResult:
    rng = _as_rng(rng)
    n_r = feat_r.shape[0]
    n_m = feat_m.shape[0]
    source = np.zeros(n_r, dtype=np.int64)
    moving_index = np.full(n_r, -1, dtype=np.int64)

    def _result(features, n_eligible, n_swapped):
        return MixResult(
            features=features,
            source=torch.from_numpy(source),
            moving_index=torch.from_numpy(moving_index),
            n_eligible=int(n_eligible),
            n_swapped=int(n_swapped),
        )

    if n_r == 0 or n_m == 0:
        return _result(feat_r.clone(), 0, 0)

    grid_r, _ = coords_to_grid(coords_r)
    grid_m, _ = coords_to_grid(coords_m)
    aligned_m = align_moving_to_reference(grid_r, grid_m)
    cell_to_m = {}
    for mi, (r, c) in enumerate(aligned_m.tolist()):
        cell_to_m.setdefault((r, c), mi)

    roi_r_np = _roi_array(roi_r, n_r)
    roi_m_np = _roi_array(roi_m, n_m)
    ref_idx, mov_idx = [], []
    for ri, (r, c) in enumerate(grid_r.tolist()):
        mi = cell_to_m.get((r, c))
        if mi is not None and (roi_r_np[ri] or roi_m_np[mi]):
            ref_idx.append(ri)
            mov_idx.append(mi)
    ref_idx = np.asarray(ref_idx, dtype=np.int64)
    mov_idx = np.asarray(mov_idx, dtype=np.int64)

    if pca_stats is not None and T_PCA < 1.0 and len(ref_idx):
        d = pca_distance_norm(pca_stats, feat_r[torch.from_numpy(ref_idx)], feat_m[torch.from_numpy(mov_idx)])
        keep = (d < T_PCA).numpy()
        ref_idx, mov_idx = ref_idx[keep], mov_idx[keep]

    n_eligible = len(ref_idx)
    target = math.ceil(P * n_eligible) if (P > 0 and n_eligible > 0) else 0
    target = min(target, n_eligible)
    if target == 0:
        return _result(feat_r.clone(), n_eligible, 0)

    N = max(int(N), 1)
    block_keys = grid_r[ref_idx] // N
    uniq, inverse = np.unique(block_keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse, minlength=len(uniq)))[:-1])
    chosen, count = [], 0
    for b in rng.permutation(len(uniq)):
        if count >= target:
            break
        m = members[b]
        need = target - count
        if len(m) > need:
            m = np.sort(rng.choice(m, size=need, replace=False))
        chosen.append(m)
        count += len(m)
    sel = np.concatenate(chosen)
    ref_sel, mov_sel = ref_idx[sel], mov_idx[sel]

    mixed = feat_r.float().clone()
    mixed[torch.from_numpy(ref_sel)] = feat_m.float()[torch.from_numpy(mov_sel)]
    source[ref_sel] = 1
    moving_index[ref_sel] = mov_sel
    return _result(mixed.to(feat_r.dtype), n_eligible, len(ref_sel))


DEFAULT_N_SCHEDULE = (16, 8, 4, 2, 1)
DEFAULT_TPCA_SCHEDULE = (1.0, 0.8, 0.6, 0.4, 0.2)
DEFAULT_P_SCHEDULE = (0.1, 0.2, 0.3, 0.4, 0.5)
SCHEDULER_NAMES = ("N", "TPCA", "P")
CURRICULUM_STRATEGIES = ("hold", "back", "fixed")


def parse_schedule(text, cast) -> tuple:
    if isinstance(text, str):
        items = [t.strip() for t in text.split(",") if t.strip()]
    elif isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [text]
    if not items:
        raise ValueError(f"empty schedule: {text!r}")
    return tuple(cast(v) for v in items)


def _parse_disabled(disabled):
    if disabled is None:
        return frozenset()
    if isinstance(disabled, str):
        disabled = [t.strip() for t in disabled.split(",") if t.strip()]
    disabled = frozenset(disabled)
    unknown = disabled - set(SCHEDULER_NAMES)
    if unknown:
        raise ValueError(f"unknown scheduler(s) {sorted(unknown)}; expected a subset of {SCHEDULER_NAMES}")
    return disabled


@dataclass
class CurriculumSchedules:
    N: tuple = DEFAULT_N_SCHEDULE
    T_PCA: tuple = DEFAULT_TPCA_SCHEDULE
    P: tuple = DEFAULT_P_SCHEDULE
    disabled: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        self.N = parse_schedule(self.N, int)
        self.T_PCA = parse_schedule(self.T_PCA, float)
        self.P = parse_schedule(self.P, float)
        self.disabled = _parse_disabled(self.disabled)
        if min(self.N) < 1:
            raise ValueError(f"N schedule must be >= 1: {self.N}")
        if min(self.T_PCA) < 0.0:
            raise ValueError(f"T_PCA schedule must be >= 0: {self.T_PCA}")
        if min(self.P) < 0.0 or max(self.P) > 1.0:
            raise ValueError(f"P schedule must lie in [0, 1]: {self.P}")

    def _named(self):
        return (("N", self.N), ("TPCA", self.T_PCA), ("P", self.P))

    @property
    def num_stages(self) -> int:
        lengths = [len(s) for name, s in self._named() if name not in self.disabled]
        return max(lengths + [1])

    def params(self, stage):
        stage = int(stage)
        if stage < 0:
            raise ValueError(f"stage must be >= 0, got {stage}")
        out = [s[0] if name in self.disabled else s[min(stage, len(s) - 1)] for name, s in self._named()]
        return int(out[0]), float(out[1]), float(out[2])


class CurriculumState:

    def __init__(self, schedules, stage=0, active=False):
        self.schedules = schedules
        self._buf = torch.zeros(2, dtype=torch.int64).share_memory_()
        self.set(stage=stage, active=active)

    @property
    def stage(self) -> int:
        return int(self._buf[0].item())

    @property
    def active(self) -> bool:
        return bool(self._buf[1].item())

    def set(self, stage=None, active=None):
        if stage is not None:
            stage = int(stage)
            if stage < 0:
                raise ValueError(f"stage must be >= 0, got {stage}")
            self._buf[0] = stage
        if active is not None:
            self._buf[1] = int(bool(active))

    def params(self):
        return self.schedules.params(self.stage)

    def __repr__(self):
        return f"CurriculumState(stage={self.stage}, active={self.active}, params={self.params()})"


class CurriculumController:

    def __init__(
        self, num_stages, strategy="hold", warmup_epochs=0, rel_drop=0.05, abs_threshold=None, min_epochs_per_stage=1
    ):
        if strategy not in CURRICULUM_STRATEGIES:
            raise ValueError(f"strategy must be one of {CURRICULUM_STRATEGIES}, got {strategy!r}")
        if int(num_stages) < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        if not 0.0 <= float(rel_drop) < 1.0:
            raise ValueError(f"rel_drop must lie in [0, 1), got {rel_drop}")
        if int(min_epochs_per_stage) < 1:
            raise ValueError(f"min_epochs_per_stage must be >= 1, got {min_epochs_per_stage}")
        if int(warmup_epochs) < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.num_stages = int(num_stages)
        self.strategy = strategy
        self.warmup_epochs = int(warmup_epochs)
        self.rel_drop = float(rel_drop)
        self.abs_threshold = None if abs_threshold is None else float(abs_threshold)
        self.min_epochs_per_stage = int(min_epochs_per_stage)

        self.stage = 0
        self.active = self.is_active(0)
        self.ref_loss = None
        self.epochs_in_stage = 0
        self.history = []

    def is_active(self, epoch) -> bool:
        return int(epoch) >= self.warmup_epochs

    def threshold(self):
        if self.abs_threshold is not None:
            return self.abs_threshold
        if self.ref_loss is None:
            return None
        return self.ref_loss * (1.0 - self.rel_drop)

    def update(self, epoch, loss):
        epoch = int(epoch)
        self.active = self.is_active(epoch + 1)
        loss = None if loss is None else float(loss)
        if loss is not None and not math.isfinite(loss):
            loss = None
        threshold = None

        if epoch >= self.warmup_epochs and loss is not None and self.strategy != "fixed":
            self.epochs_in_stage += 1
            if self.ref_loss is None:
                self.ref_loss = loss
            elif self.epochs_in_stage >= self.min_epochs_per_stage:
                threshold = self.threshold()
                if loss < threshold:
                    new_stage = min(self.stage + 1, self.num_stages - 1)
                elif self.strategy == "back":
                    new_stage = max(self.stage - 1, 0)
                else:
                    new_stage = self.stage
                if new_stage != self.stage:
                    self.stage = new_stage
                    self.ref_loss = None
                    self.epochs_in_stage = 0

        self.history.append(
            {"epoch": epoch, "loss": loss, "threshold": threshold, "stage": self.stage, "active": self.active}
        )
        return self.stage, self.active

    def state_dict(self):
        return {
            "stage": self.stage,
            "active": self.active,
            "ref_loss": self.ref_loss,
            "epochs_in_stage": self.epochs_in_stage,
            "history": [dict(h) for h in self.history],
        }

    def load_state_dict(self, state):
        self.stage = min(max(int(state["stage"]), 0), self.num_stages - 1)
        self.active = bool(state["active"])
        self.ref_loss = None if state.get("ref_loss") is None else float(state["ref_loss"])
        self.epochs_in_stage = int(state.get("epochs_in_stage", 0))
        self.history = [dict(h) for h in state.get("history", [])]


SOFT_LABEL_METHODS = ("linear", "random", "vlm")


def _label_missing(y) -> bool:
    if y is None:
        return True
    y = float(y)
    return math.isnan(y) or y < 0


def _value_missing(v) -> bool:
    return v is None or math.isnan(float(v))


def _source_weights(f, method, rng, w_r=None, w_m=None):
    if method not in SOFT_LABEL_METHODS:
        raise ValueError(f"label method must be one of {SOFT_LABEL_METHODS}, got {method!r}")
    if method == "random":
        return (1.0, 0.0) if _as_rng(rng).random() < 0.5 else (0.0, 1.0)
    if method == "vlm" and w_r is not None and w_m is not None and (w_r + w_m) > 0:
        total = float(w_r) + float(w_m)
        return float(w_r) / total, float(w_m) / total
    f = min(max(float(f), 0.0), 1.0)
    return f, 1.0 - f


def _onehot(y, num_classes):
    y = int(y)
    if y >= num_classes:
        raise ValueError(f"class id {y} out of range for {num_classes} classes")
    t = torch.zeros(num_classes, dtype=torch.float32)
    t[y] = 1.0
    return t


def soft_cls_target(
    y_r: Optional[int], y_m: Optional[int], num_classes: int, f: float, method: str, rng=None, w_r=None, w_m=None
) -> torch.Tensor:
    C = int(num_classes)
    if method not in SOFT_LABEL_METHODS:
        raise ValueError(f"label method must be one of {SOFT_LABEL_METHODS}, got {method!r}")
    if _label_missing(y_r):
        return torch.full((C,), -1.0, dtype=torch.float32)
    oh_r = _onehot(y_r, C)
    if _label_missing(y_m):
        return oh_r
    oh_m = _onehot(y_m, C)
    a_r, a_m = _source_weights(f, method, rng, w_r, w_m)
    return a_r * oh_r + a_m * oh_m


def soft_reg_target(v_r, v_m, f: float, method: str, rng=None, w_r=None, w_m=None) -> float:
    if method not in SOFT_LABEL_METHODS:
        raise ValueError(f"label method must be one of {SOFT_LABEL_METHODS}, got {method!r}")
    if _value_missing(v_r):
        return -1.0
    if _value_missing(v_m):
        return float(v_r)
    a_r, a_m = _source_weights(f, method, rng, w_r, w_m)
    return a_r * float(v_r) + a_m * float(v_m)


def vlm_evidence_weights(scores_r, scores_m, src, ref_idx_sel, mov_idx_sel, y_r, y_m):
    if scores_r is None or scores_m is None or _label_missing(y_r) or _label_missing(y_m):
        return None, None
    src = torch.as_tensor(src).long().reshape(-1)
    ref_idx_sel = torch.as_tensor(ref_idx_sel).long().reshape(-1)
    mov_idx_sel = torch.as_tensor(mov_idx_sel).long().reshape(-1)
    from_r, from_m = src == 0, src == 1
    w_r = float(torch.as_tensor(scores_r).float()[ref_idx_sel[from_r], int(y_r)].sum())
    w_m = float(torch.as_tensor(scores_m).float()[mov_idx_sel[from_m], int(y_m)].sum())
    return w_r, w_m


def cls_soft_valid(target) -> torch.Tensor:
    return target[:, 0] >= 0


@dataclass
class SlideMixConfig:

    state: CurriculumState
    prob: float = 0.5
    label_method: str = "vlm"
    pca_stats: Optional[PCAStats] = None
    roi_dir: Optional[Path] = None
    vlm_dir: Optional[Path] = None

    def __post_init__(self):
        self.prob = float(self.prob)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must lie in [0, 1], got {self.prob}")
        if self.label_method not in SOFT_LABEL_METHODS:
            raise ValueError(f"label_method must be one of {SOFT_LABEL_METHODS}, got {self.label_method!r}")
        self.roi_dir = None if self.roi_dir is None else Path(self.roi_dir)
        self.vlm_dir = None if self.vlm_dir is None else Path(self.vlm_dir)
