import importlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.spatial import cKDTree

from .slidemix import infer_stride

CONCH_MODULE = "models.CONCH.open_clip_custom"
CONCH_MODEL_CFG = "conch_ViT-B-16"
CONCH_HF_PRETRAINED = "hf_hub:MahmoodLab/conch"
CONCH_CONTEXT_LENGTH = 128
MAX_CONTEXT_CHARS = 300

CONCH_TEMPLATES = [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
]


def _coords_np(coords_yx) -> np.ndarray:
    if isinstance(coords_yx, torch.Tensor):
        coords_yx = coords_yx.detach().cpu().numpy()
    return np.asarray(coords_yx).reshape(-1, 2)


def tile_centers(coords_yx, stride=None) -> np.ndarray:
    coords = _coords_np(coords_yx).astype(np.float64)
    if stride is None:
        stride = infer_stride(coords.astype(np.int64)) if len(coords) else 1
    return coords + float(stride) / 2.0


def map_tiles_nearest(src_coords, dst_coords, src_stride=None, dst_stride=None, max_dist=None) -> np.ndarray:
    src = _coords_np(src_coords)
    dst = _coords_np(dst_coords)
    if len(dst) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(src) == 0:
        return np.full(len(dst), -1, dtype=np.int64)
    src_stride = infer_stride(src) if src_stride is None else src_stride
    dst_stride = infer_stride(dst) if dst_stride is None else dst_stride
    if max_dist is None:
        max_dist = float(max(src_stride, dst_stride))
    tree = cKDTree(tile_centers(src, src_stride))
    dist, idx = tree.query(tile_centers(dst, dst_stride), k=1)
    idx = np.asarray(idx, dtype=np.int64)
    idx[np.asarray(dist) > max_dist] = -1
    return idx


def load_slide_tensors(path, keys=("features", "coords_yx")) -> dict:
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        available = set(f.keys())
        for k in keys:
            if k in available:
                t = f.get_tensor(k)
                out[k] = t.float() if k == "features" else t
    return out


def load_slide_metadata(path) -> dict:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def save_slide_tensors(path, tensors: dict, metadata: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = None
    if metadata:
        md = {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items()}
    data = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    tmp = path.with_name(path.name + ".tmp")
    try:
        save_file(data, str(tmp), metadata=md)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def embedding_dir(data_path, dataset_name, embedding_name) -> Path:
    from .CONSTS import EMBEDDING_MODELS

    return Path(data_path) / dataset_name / EMBEDDING_MODELS[embedding_name]["folder_name"]


def list_slides(directory) -> list:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.safetensors"))


def _load_conch(model_cfg, pretrained, device, hf_token):
    conch = importlib.import_module(CONCH_MODULE)
    model = conch.create_model_from_pretrained(
        model_cfg, pretrained, device=device, return_transform=False, hf_auth_token=hf_token
    )
    return model, conch.get_tokenizer(), conch.tokenize


class ConchTextEncoder:

    def __init__(self, device="cpu", checkpoint=None, hf_token_env="HF_TOKEN", batch_size=64, loader=None):
        self.device = device
        self.model_cfg = CONCH_MODEL_CFG
        self.pretrained = checkpoint or CONCH_HF_PRETRAINED
        self.hf_token_env = hf_token_env
        self.batch_size = int(batch_size)
        self._loader = loader or _load_conch
        self._model = None
        self._tokenizer = None
        self._tokenize = None

    def _ensure_loaded(self):
        if self._model is None:
            hf_token = os.environ.get(self.hf_token_env) if self.hf_token_env else None
            model, tokenizer, tokenize = self._loader(self.model_cfg, self.pretrained, self.device, hf_token)
            self._model = model.eval().to(self.device)
            self._tokenizer, self._tokenize = tokenizer, tokenize

    def encode(self, texts) -> torch.Tensor:
        self._ensure_loaded()
        texts = list(texts)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self._tokenize(self._tokenizer, texts[start : start + self.batch_size]).to(self.device)
                out.append(self._model.encode_text(tokens, normalize=True).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, 512)

    @property
    def logit_scale(self) -> float:
        self._ensure_loaded()
        return float(self._model.logit_scale.exp().item())


def _as_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _short_context(context: Optional[str]) -> str:
    if not context:
        return ""
    ctx = " ".join(str(context).split()).rstrip(". ")
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0]
    return ctx


def class_text_embeddings(encoder, class_names, templates=CONCH_TEMPLATES, context: Optional[str] = None) -> torch.Tensor:
    ctx = _short_context(context)
    rows = []
    for cls in class_names:
        names = [cls] if isinstance(cls, str) else list(cls)
        texts = [t.replace("CLASSNAME", n) for n in names for t in templates]
        if ctx:
            texts = [f"{ctx}. {t}" for t in texts]
        emb = F.normalize(_as_float_tensor(encoder.encode(texts)), dim=-1)
        rows.append(F.normalize(emb.mean(dim=0), dim=-1))
    return torch.stack(rows)


def tile_class_probs(tile_emb, class_emb, logit_scale) -> torch.Tensor:
    tile = F.normalize(_as_float_tensor(tile_emb), dim=-1)
    cls = F.normalize(_as_float_tensor(class_emb), dim=-1)
    return torch.softmax(float(logit_scale) * (tile @ cls.T), dim=-1)


def text_tile_similarity(tile_emb, text_emb) -> torch.Tensor:
    tile = F.normalize(_as_float_tensor(tile_emb), dim=-1)
    text = F.normalize(_as_float_tensor(text_emb).reshape(-1), dim=0)
    return tile @ text
