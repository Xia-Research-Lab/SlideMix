import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from tools import slidemix_rag as rag  # noqa: E402
from utils import slidemix_vlm as vlm  # noqa: E402

DATASET_CONFIG_DIR = Path(__file__).resolve().parents[1] / "dataset_configs"
SUMMARY_NAME = "vlm_scores_summary.json"

TEXT_ENCODERS = {"conch": vlm.ConchTextEncoder}

_GLEASON_RE = re.compile(r"^\d\+\d$")

BUILTIN_CLASS_NAMES = {
    "isup_grade": {
        "0": ["benign prostate tissue", "ISUP grade group 0"],
        "1": ["ISUP grade group 1 prostate adenocarcinoma", "Gleason score 3+3 prostate adenocarcinoma"],
        "2": ["ISUP grade group 2 prostate adenocarcinoma", "Gleason score 3+4 prostate adenocarcinoma"],
        "3": ["ISUP grade group 3 prostate adenocarcinoma", "Gleason score 4+3 prostate adenocarcinoma"],
        "4": ["ISUP grade group 4 prostate adenocarcinoma", "Gleason score 8 prostate adenocarcinoma"],
        "5": ["ISUP grade group 5 prostate adenocarcinoma", "Gleason score 9-10 prostate adenocarcinoma"],
    },
    "gleason_score": {
        "0+0": ["benign prostate tissue", "Gleason score 0+0"],
        "negative": ["benign prostate tissue", "negative for prostate cancer"],
    },
    "BREAST_METASTASIS": {
        "normal": ["normal lymph node tissue"],
        "tumor": ["metastatic breast carcinoma in lymph node", "tumor"],
        "negative": ["normal lymph node tissue", "lymph node negative for metastasis"],
        "itc": ["isolated tumor cells of breast carcinoma in lymph node"],
        "micro": ["micrometastasis of breast carcinoma in lymph node"],
        "macro": ["macrometastasis of breast carcinoma in lymph node"],
    },
}


def _readable(text) -> str:
    return " ".join(str(text).replace("_", " ").split())


def load_task_config(dataset_name) -> dict:
    import yaml

    with open(DATASET_CONFIG_DIR / dataset_name / "task_configs.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def _cls_tasks(cfg) -> dict:
    types = cfg.get("all_task_dict") or {}
    labels = cfg.get("label_dict") or {}
    return {t: dict(labels[t]) for t, kind in types.items() if kind == "cls" and labels.get(t)}


def all_cls_tasks(dataset_name) -> dict:
    return _cls_tasks(load_task_config(dataset_name))


def default_cls_tasks(dataset_name) -> dict:
    cfg = load_task_config(dataset_name)
    available = _cls_tasks(cfg)
    selected = {t: available[t] for t in (cfg.get("tasks_to_run") or []) if t in available}
    if selected:
        return selected
    print(
        f"Note: no cls task with a label_dict in {dataset_name} tasks_to_run; "
        f"scoring all cls tasks with a label_dict: {list(available)}"
    )
    return available


def resolve_tasks(dataset_name, tasks=None) -> dict:
    if isinstance(tasks, dict):
        return {t: dict(ld) for t, ld in tasks.items()}
    if tasks is None:
        return default_cls_tasks(dataset_name)
    available = all_cls_tasks(dataset_name)
    names = tasks.split("%") if isinstance(tasks, str) else list(tasks)
    missing = [t for t in names if t not in available]
    if missing:
        raise ValueError(f"{missing} are not cls tasks with a label_dict in {dataset_name}; available: {sorted(available)}")
    return {t: available[t] for t in names}


def load_class_prompts(path) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object task -> {{label: [synonyms]}}")
    return data


def _synonyms(value) -> list:
    out = [value] if isinstance(value, str) else [str(v) for v in value]
    out = [s for s in out if s.strip()]
    if not out:
        raise ValueError(f"empty synonym list: {value!r}")
    return out


def _default_names(task_name, label) -> list:
    builtin = BUILTIN_CLASS_NAMES.get(task_name, {})
    if label in builtin:
        return list(builtin[label])
    if _GLEASON_RE.match(label):
        return [f"Gleason pattern {label} prostate adenocarcinoma"]
    readable = _readable(label)
    if re.search(r"[A-Za-z]", readable):
        return [readable]
    return [f"{readable} ({_readable(task_name)})"]


def class_names_for_task(task_name, label_dict_for_task, class_prompts=None) -> list:
    if isinstance(class_prompts, (str, Path)):
        class_prompts = load_class_prompts(class_prompts)
    overrides = {str(k): v for k, v in ((class_prompts or {}).get(task_name) or {}).items()}
    by_index = {}
    for label, idx in label_dict_for_task.items():
        label = str(label)
        names = _synonyms(overrides[label]) if label in overrides else _default_names(task_name, label)
        by_index.setdefault(int(idx), [])
        by_index[int(idx)] += [n for n in names if n not in by_index[int(idx)]]
    C = len(by_index)
    if sorted(by_index) != list(range(C)):
        raise ValueError(f"{task_name}: class indices must be 0..C-1, got {sorted(by_index)}")
    return [by_index[i] for i in range(C)]


def class_order(label_dict_for_task) -> list:
    by_index = {}
    for label, idx in label_dict_for_task.items():
        by_index.setdefault(int(idx), []).append(str(label))
    return ["|".join(by_index[i]) for i in sorted(by_index)]


def task_context(dataset_name, task_name) -> str:
    from utils.CONSTS import DATASETS_ORGAN

    organ = DATASETS_ORGAN.get(dataset_name, _readable(dataset_name))
    return f"{organ}, H&E, {_readable(task_name)}"


def rag_context(retriever, L, P, k) -> tuple:
    docs = retriever.query(L, P, k)
    pmids = [str(d.get("pmid")) for d in docs]
    titles_only = [{"pmid": d.get("pmid"), "title": d.get("title", ""), "abstract": ""} for d in docs]
    fmt = getattr(retriever, "format_context", None) or rag.format_context
    text = fmt(titles_only, max_chars=vlm.MAX_CONTEXT_CHARS)
    lines = [ln.strip()[2:] if ln.strip().startswith("- ") else ln.strip() for ln in (text or "").splitlines()]
    context = "; ".join(ln for ln in lines if ln)
    return (context or None), pmids


def build_task_class_embeddings(encoder, dataset_name, tasks, retriever=None, rag_k=3, class_prompts=None) -> dict:
    out = {}
    for task, label_dict in tasks.items():
        names = class_names_for_task(task, label_dict, class_prompts)
        context, pmids = None, []
        if retriever is not None:
            P = f"histopathologic features of {', '.join(n[0] for n in names)}"
            context, pmids = rag_context(retriever, task_context(dataset_name, task), P, rag_k)
        emb = vlm.class_text_embeddings(encoder, names, vlm.CONCH_TEMPLATES, context).float()
        out[task] = (emb, context, pmids)
    return out


def score_slide(conch_feats, conch_coords, giga_coords, task_embs, logit_scale, conch_stride=None, giga_stride=None) -> dict:
    feats = vlm._as_float_tensor(conch_feats)
    n_c = vlm._coords_np(conch_coords).shape[0]
    if feats.shape[0] != n_c:
        raise ValueError(f"CONCH features ({feats.shape[0]}) and coords ({n_c}) differ in length")
    idx = torch.from_numpy(vlm.map_tiles_nearest(conch_coords, giga_coords, conch_stride, giga_stride))
    mapped = idx >= 0
    out = {}
    for task, value in task_embs.items():
        class_emb = vlm._as_float_tensor(value[0] if isinstance(value, (tuple, list)) else value)
        C = class_emb.shape[0]
        scores = torch.full((idx.shape[0], C), 1.0 / C, dtype=torch.float32)
        if n_c and bool(mapped.any()):
            if feats.shape[1] != class_emb.shape[1]:
                raise ValueError(f"CONCH tile dim {feats.shape[1]} != text dim {class_emb.shape[1]} ({task})")
            probs = vlm.tile_class_probs(feats, class_emb, logit_scale).float()
            scores[mapped] = probs[idx[mapped]]
        out[task] = scores.contiguous()
    return out


def run(
    data_path,
    dataset_name,
    out_dir,
    encoder,
    tasks=None,
    conch_embedding_name="CONCH",
    giga_embedding_name="GigaPath",
    retriever=None,
    rag_k=3,
    class_prompts=None,
    slides=None,
    limit=None,
    overwrite=False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = resolve_tasks(dataset_name, tasks)
    if not tasks:
        raise ValueError(f"no cls tasks with a label_dict to score for {dataset_name}")
    if isinstance(class_prompts, (str, Path)):
        class_prompts = load_class_prompts(class_prompts)

    giga_dir = vlm.embedding_dir(data_path, dataset_name, giga_embedding_name)
    conch_dir = vlm.embedding_dir(data_path, dataset_name, conch_embedding_name)
    slide_ids = vlm.list_slides(giga_dir)
    if slides is not None:
        wanted = set(str(s) for s in slides)
        slide_ids = [s for s in slide_ids if s in wanted]
    if limit is not None:
        slide_ids = slide_ids[: int(limit)]

    task_embs = build_task_class_embeddings(encoder, dataset_name, tasks, retriever, rag_k, class_prompts)
    logit_scale = float(encoder.logit_scale)
    order = {t: class_order(ld) for t, ld in tasks.items()}
    names = {t: class_names_for_task(t, ld, class_prompts) for t, ld in tasks.items()}
    rag_info = {t: {"context": ctx, "pmids": pmids} for t, (_e, ctx, pmids) in task_embs.items()}
    metadata = {
        "tasks": list(tasks),
        "class_order": order,
        "class_names": names,
        "rag_pmids": {t: v["pmids"] for t, v in rag_info.items()},
        "rag_context": {t: v["context"] for t, v in rag_info.items()},
        "logit_scale": logit_scale,
        "conch_embedding_name": conch_embedding_name,
        "embedding_name": giga_embedding_name,
    }

    done, skipped = [], {}
    for i, sid in enumerate(slide_ids):
        out_path = out_dir / f"{sid}.safetensors"
        conch_path = conch_dir / f"{sid}.safetensors"
        if out_path.exists() and not overwrite:
            skipped[sid] = "exists"
            continue
        if not conch_path.exists():
            skipped[sid] = f"missing CONCH embeddings ({conch_path})"
            continue
        try:
            giga = vlm.load_slide_tensors(giga_dir / f"{sid}.safetensors", keys=("coords_yx",))
            conch = vlm.load_slide_tensors(conch_path, keys=("features", "coords_yx"))
            for name, d, keys in (("GigaPath", giga, ("coords_yx",)), ("CONCH", conch, ("features", "coords_yx"))):
                absent = [k for k in keys if k not in d]
                if absent:
                    raise KeyError(f"{name} file lacks {absent}")
            scores = score_slide(conch["features"], conch["coords_yx"], giga["coords_yx"], task_embs, logit_scale)
            vlm.save_slide_tensors(out_path, scores, metadata=metadata)
        except Exception as e:
            skipped[sid] = f"error: {type(e).__name__}: {e}"
            print(f"[{i + 1}/{len(slide_ids)}] {sid}: {skipped[sid]}")
            continue
        done.append(sid)
        print(f"[{i + 1}/{len(slide_ids)}] {sid}: {giga['coords_yx'].shape[0]} tiles, tasks {list(scores)}")

    summary = {
        "dataset_name": dataset_name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "n_slides": len(slide_ids),
        "n_done": len(done),
        "n_skipped": len(skipped),
        "done": done,
        "skipped": skipped,
        "tasks": list(tasks),
        "class_order": order,
        "class_names": names,
        "rag_used": retriever is not None,
        "rag": rag_info,
        "logit_scale": logit_scale,
        "conch_embedding_name": conch_embedding_name,
        "embedding_name": giga_embedding_name,
    }
    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def make_text_encoder(name, device="cpu", checkpoint=None):
    if name not in TEXT_ENCODERS:
        raise ValueError(f"unknown text encoder {name!r}; available: {sorted(TEXT_ENCODERS)}")
    return TEXT_ENCODERS[name](device=device, checkpoint=checkpoint)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SlideMix VLM soft-labeler: per-tile CONCH class probabilities")
    p.add_argument("--data_path", type=str, required=True, help="root holding <dataset_name>/<embedding folders>")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="output folder (pass to main.py --slidemix_vlm_dir)")
    p.add_argument("--tasks", type=str, default=None, help="cls tasks split by %% (default: cls tasks of tasks_to_run in task_configs.yaml, like main.py; all cls tasks if none qualify)")
    p.add_argument("--conch_embedding_name", type=str, default="CONCH", help="CONSTS.EMBEDDING_MODELS key of CONCH tiles")
    p.add_argument("--embedding_name", type=str, default="GigaPath", help="training embedding (defines tile order)")
    p.add_argument("--text_encoder", type=str, default="conch", help=f"one of {sorted(TEXT_ENCODERS)}")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--conch_checkpoint", type=str, default=None, help="local CONCH weights (default: HF hub, env HF_TOKEN)")
    p.add_argument("--rag_index", type=str, default=None, help="slidemix_rag.py build output folder")
    p.add_argument("--rag_encoder", type=str, default="biomedclip", help="slidemix_rag.ENCODERS key (query encoder)")
    p.add_argument("--rag_k", type=int, default=3)
    p.add_argument("--no_rag", action="store_true", help="no retrieved context (VLM/u ablation)")
    p.add_argument("--class_prompts", type=str, default=None, help="JSON {task: {label: [synonyms]}} overriding names")
    p.add_argument("--slides", type=str, nargs="*", default=None, help="only these slide ids")
    p.add_argument("--limit", type=int, default=None, help="at most this many slides")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.text_encoder not in TEXT_ENCODERS:
        parser.error(f"unknown --text_encoder {args.text_encoder!r}; available: {sorted(TEXT_ENCODERS)}")
    retriever = None
    if not args.no_rag:
        if args.rag_index is None:
            parser.error("--rag_index is required unless --no_rag (VLM/u ablation)")
        if args.rag_encoder not in rag.ENCODERS:
            parser.error(f"unknown --rag_encoder {args.rag_encoder!r}; available: {sorted(rag.ENCODERS)}")
        retriever = rag.Retriever(args.rag_index, rag.ENCODERS[args.rag_encoder](device=args.device))
    encoder = make_text_encoder(args.text_encoder, device=args.device, checkpoint=args.conch_checkpoint)
    summary = run(
        args.data_path,
        args.dataset_name,
        args.out_dir,
        encoder,
        tasks=args.tasks,
        conch_embedding_name=args.conch_embedding_name,
        giga_embedding_name=args.embedding_name,
        retriever=retriever,
        rag_k=args.rag_k,
        class_prompts=args.class_prompts,
        slides=args.slides,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        f"VLM scores: {summary['n_done']}/{summary['n_slides']} slides -> {args.out_dir} "
        f"({summary['n_skipped']} skipped, RAG {'on' if retriever is not None else 'off'})"
    )
    return summary


if __name__ == "__main__":
    main()
