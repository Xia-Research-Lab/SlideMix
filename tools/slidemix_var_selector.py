import argparse
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from utils.slidemix_vlm import (  # noqa: E402
    ConchTextEncoder,
    embedding_dir,
    list_slides,
    load_slide_tensors,
    map_tiles_nearest,
    save_slide_tensors,
    text_tile_similarity,
)

DEFAULT_REGION_PROMPT = "Identify key pathological regions in this WSI."
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TOP_FRAC = 0.25
DEFAULT_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
WSI_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".vms")
DATASET_CONFIG_DIR = Path(__file__).resolve().parents[1] / "dataset_configs"
MAX_GROUNDING_CHARS = 600
MAX_LABELS_PER_TASK = 12
SUMMARY_NAME = "var_summary.json"

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "regions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["description", "regions"],
}

TEXT_ENCODERS = {"conch": ConchTextEncoder}


def task_context(dataset_name, tasks=None, staining="H&E", config_dir=DATASET_CONFIG_DIR) -> str:
    try:
        from utils.CONSTS import DATASETS_ORGAN

        organ = DATASETS_ORGAN.get(dataset_name, dataset_name)
    except ImportError:
        organ = dataset_name
    ctx = f"{organ}, {staining} staining"

    cfg = {}
    cfg_path = Path(config_dir) / str(dataset_name) / "task_configs.yaml"
    if cfg_path.exists():
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if tasks is None:
        tasks = cfg.get("tasks_to_run", list(cfg.get("all_task_dict", {})))
    parts = []
    for task in tasks or []:
        labels = [str(k) for k in (cfg.get("label_dict", {}) or {}).get(task, {}) or {}]
        if len(labels) > MAX_LABELS_PER_TASK:
            labels = labels[:MAX_LABELS_PER_TASK] + ["..."]
        parts.append(f"{task} ({', '.join(labels)})" if labels else str(task))
    if parts:
        ctx += ", tasks: " + ", ".join(parts)
    return ctx


def threshold_mask(scores, tau=None, top_frac=None, min_tiles=1) -> np.ndarray:
    if tau is not None and top_frac is not None:
        raise ValueError("give either tau or top_frac, not both")
    if tau is None and top_frac is None:
        top_frac = DEFAULT_TOP_FRAC
    if top_frac is not None and not (0.0 <= float(top_frac) <= 1.0):
        raise ValueError(f"top_frac must be in [0, 1], got {top_frac}")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = np.where(np.isnan(s), -np.inf, s)
    n = len(s)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.argsort(-s, kind="stable")
    if tau is not None:
        mask = s > float(tau)
    else:
        top = order[: min(n, math.ceil(float(top_frac) * n))]
        mask[top[s[top] > -np.inf]] = True
    m = min(max(int(min_tiles), 0), n)
    if mask.sum() < m:
        mask[order[:m]] = True
    return mask


def _trim(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return text


def ground_description(text_encoder, description, conch_feats, conch_coords, giga_coords, conch_stride=None, giga_stride=None) -> np.ndarray:
    text = _trim(description, MAX_GROUNDING_CHARS)
    text_emb = text_encoder.encode([text])
    text_emb = text_emb[0] if len(text_emb) else text_emb
    sims = text_tile_similarity(conch_feats, text_emb).numpy().astype(np.float64)
    idx = map_tiles_nearest(conch_coords, giga_coords, src_stride=conch_stride, dst_stride=giga_stride)
    out = np.full(len(idx), -np.inf, dtype=np.float64)
    valid = idx >= 0
    out[valid] = sims[idx[valid]]
    return out


def build_gemini_prompt(task_ctx: str, region_prompt: str, rag_context: str) -> str:
    literature = rag_context.strip() if rag_context and rag_context.strip() else "(no literature retrieved)"
    return (
        "You are assisting computational pathology research. The attached image is a low-magnification "
        "thumbnail of a whole slide image (WSI).\n"
        f"Task context: {task_ctx or 'not specified'}\n"
        f"Instruction: {region_prompt}\n"
        "Relevant literature (PubMed titles and abstract snippets):\n"
        f"{literature}\n\n"
        "Describe the diagnostically relevant tissue regions visible in this slide for the task above. "
        "For each region give its morphology (e.g. dense cellular clusters, glandular structures, necrosis, "
        "stroma) and its location on the slide (e.g. upper left, lower center), for example: "
        "\"dense cellular clusters in the upper left and glandular structures in the lower center\". "
        "Ground the description in the literature above where it applies. Do not give a diagnosis, grade "
        "or label prediction; describe only the regions.\n"
        "Answer with JSON only, exactly of the form "
        '{"description": "<one or two sentences>", "regions": ["<short region phrase>", ...]}.'
    )


def parse_gemini_json(text) -> dict:
    raw = (text or "").strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", raw, flags=re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini response is not JSON: {raw[:200]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"Gemini response is not a JSON object: {raw[:200]!r}")
    regions = obj.get("regions") or []
    if isinstance(regions, str):
        regions = [regions]
    regions = [str(r).strip() for r in regions if str(r).strip()]
    description = str(obj.get("description") or "").strip()
    if not description:
        description = "; ".join(regions)
    if not description:
        raise ValueError("Gemini response has neither description nor regions")
    return {"description": description, "regions": regions}


class GeminiRegionDescriber:

    def __init__(
        self,
        model=DEFAULT_GEMINI_MODEL,
        api_key_env=DEFAULT_API_KEY_ENVS,
        client=None,
        cache_path=None,
        temperature=0.2,
        max_retries=3,
        backoff=2.0,
        sleep=time.sleep,
        types_module=None,
    ):
        self.model = model
        self.api_key_env = (api_key_env,) if isinstance(api_key_env, str) else tuple(api_key_env)
        self._client = client
        self._types = types_module
        self.cache_path = Path(cache_path) if cache_path else None
        self.temperature = float(temperature)
        self.max_retries = max(1, int(max_retries))
        self.backoff = float(backoff)
        self._sleep = sleep
        self.cache = {}
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _api_key(self) -> str:
        for name in self.api_key_env:
            value = os.environ.get(name)
            if value:
                return value
        raise RuntimeError(f"No Gemini API key: set one of the environment variables {', '.join(self.api_key_env)}")

    @property
    def client(self):
        if self._client is None:
            api_key = self._api_key()
            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    @property
    def types(self):
        if self._types is None:
            from google.genai import types

            self._types = types
        return self._types

    def build_contents(self, thumbnail_png: bytes, prompt_text: str) -> list:
        return [self.types.Part.from_bytes(data=thumbnail_png, mime_type="image/png"), prompt_text]

    def build_config(self):
        return self.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=self.temperature,
        )

    def _save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    def describe(self, slide_id, thumbnail_png: bytes, task_ctx: str, region_prompt: str, rag_context: str) -> dict:
        cached = self.cache.get(str(slide_id))
        if cached is not None:
            return {"description": cached["description"], "regions": list(cached.get("regions", []))}
        prompt_text = build_gemini_prompt(task_ctx, region_prompt, rag_context)
        client = self.client
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=self.build_contents(thumbnail_png, prompt_text),
                    config=self.build_config(),
                )
                result = parse_gemini_json(response.text)
                break
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    self._sleep(self.backoff * (2**attempt))
        else:
            raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last_err}") from last_err
        self.cache[str(slide_id)] = {**result, "model": self.model}
        self._save_cache()
        return result


def find_wsi(wsi_dir, slide_id, exts=WSI_EXTENSIONS) -> Optional[Path]:
    if wsi_dir is None:
        return None
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.is_dir():
        return None
    exts = tuple(e.lower() for e in exts)
    for ext in exts:
        p = wsi_dir / f"{slide_id}{ext}"
        if p.is_file():
            return p
    for p in sorted(wsi_dir.rglob(f"{slide_id}.*")):
        if p.is_file() and p.stem == str(slide_id) and p.suffix.lower() in exts:
            return p
    return None


def read_thumbnail_png(wsi_path, max_side=1024) -> bytes:
    import openslide
    from PIL import Image

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        level = slide.level_count - 1
        size = tuple(int(v) for v in slide.level_dimensions[level])
        region = slide.read_region((0, 0), level, size)
    finally:
        slide.close()
    if region.mode == "RGBA":
        img = Image.new("RGB", region.size, (255, 255, 255))
        img.paste(region, mask=region.split()[3])
    else:
        img = region.convert("RGB")
    img.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _conch_description(task_ctx, region_prompt, docs) -> str:
    prompt = (region_prompt or "").strip()
    ctx = (task_ctx or "").strip().rstrip(".")
    text = f"{ctx}. {prompt}" if ctx else prompt
    titles = [str(d.get("title", "")).strip().rstrip(".") for d in docs or [] if str(d.get("title", "")).strip()]
    if titles:
        text = text.rstrip(".") + ". Related literature: " + "; ".join(titles) + "."
    return _trim(text, MAX_GROUNDING_CHARS)


def _gemini_text(result: dict) -> str:
    text = result["description"].strip()
    regions = result.get("regions") or []
    if regions:
        text = text.rstrip(".") + ". Regions: " + "; ".join(regions) + "."
    return text


def select_roi_for_slide(
    slide_id,
    mode,
    giga_path,
    out_dir,
    *,
    conch_path=None,
    text_encoder=None,
    describer=None,
    retriever=None,
    wsi_dir=None,
    task_ctx="",
    region_prompt=DEFAULT_REGION_PROMPT,
    rag_k=5,
    tau=None,
    top_frac=None,
    min_tiles=1,
    overwrite=False,
    thumbnail_max_side=1024,
) -> dict:
    if mode not in ("all", "conch", "gemini"):
        raise ValueError(f"unknown mode {mode!r}; use all / conch / gemini")
    if mode in ("conch", "gemini") and text_encoder is None:
        raise ValueError(f"mode {mode!r} needs a text_encoder")
    if mode == "gemini" and describer is None:
        raise ValueError("mode 'gemini' needs a describer")
    if tau is None and top_frac is None:
        top_frac = DEFAULT_TOP_FRAC

    out_path = Path(out_dir) / f"{slide_id}.safetensors"
    summary = {
        "slide_id": str(slide_id),
        "mode": mode,
        "effective_mode": mode,
        "status": "written",
        "fallback": None,
        "n_tiles": None,
        "n_selected": None,
        "description": "",
        "pmids": [],
        "out_path": str(out_path),
    }
    if out_path.exists() and not overwrite:
        summary["status"] = "skipped"
        summary["effective_mode"] = None
        return summary

    giga_coords = load_slide_tensors(giga_path, keys=("coords_yx",))["coords_yx"]
    n_g = int(giga_coords.shape[0])
    effective = mode
    fallbacks = []
    if mode != "all" and (conch_path is None or not Path(conch_path).exists()):
        effective = "all"
        fallbacks.append(f"CONCH embedding file missing ({conch_path}) -> all tiles")

    description, docs = "", []
    if effective == "gemini":
        try:
            if retriever is not None:
                docs = retriever.query(task_ctx, region_prompt, k=rag_k)
            rag_ctx = retriever.format_context(docs) if retriever is not None else ""
            wsi_path = find_wsi(wsi_dir, slide_id)
            if wsi_path is None:
                raise FileNotFoundError(f"WSI not found for {slide_id} in {wsi_dir}")
            png = read_thumbnail_png(wsi_path, max_side=thumbnail_max_side)
            result = describer.describe(slide_id, png, task_ctx, region_prompt, rag_ctx)
            description = _gemini_text(result)
        except Exception as e:
            effective = "conch"
            fallbacks.append(f"gemini failed ({type(e).__name__}: {e}) -> conch description")
    if effective == "conch":
        if retriever is not None and not docs:
            try:
                docs = retriever.query(task_ctx, region_prompt, k=rag_k)
            except Exception as e:
                fallbacks.append(f"retrieval failed ({type(e).__name__}: {e}) -> no literature")
                docs = []
        description = _conch_description(task_ctx, region_prompt, docs)

    if effective == "all":
        mask = np.ones(n_g, dtype=bool)
    else:
        conch = load_slide_tensors(conch_path, keys=("features", "coords_yx"))
        scores = ground_description(text_encoder, description, conch["features"], conch["coords_yx"], giga_coords)
        mask = threshold_mask(scores, tau=tau, top_frac=None if tau is not None else top_frac, min_tiles=min_tiles)

    pmids = [str(d.get("pmid")) for d in docs]
    n_sel = int(mask.sum())
    summary.update(
        effective_mode=effective,
        fallback="; ".join(fallbacks) if fallbacks else None,
        n_tiles=n_g,
        n_selected=n_sel,
        description=description,
        pmids=pmids,
    )
    metadata = {
        "mode": effective,
        "requested_mode": mode,
        "description": description,
        "tau": tau,
        "top_frac": None if tau is not None else top_frac,
        "n_selected": n_sel,
        "n_tiles": n_g,
        "pmids": pmids,
        "fallback": summary["fallback"],
    }
    save_slide_tensors(out_path, {"roi_mask": torch.from_numpy(mask)}, metadata=metadata)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SlideMix VAR selector: per-slide ROI masks")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--embedding_name", type=str, default="GigaPath", help="embeddings the mask is aligned to")
    p.add_argument("--conch_embedding_name", type=str, default="CONCH")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="conch", choices=["all", "conch", "gemini"])
    p.add_argument("--wsi_dir", type=str, default=None, help="raw WSIs (gemini mode thumbnails)")
    p.add_argument("--rag_index", type=str, default=None, help="index dir built by slidemix_rag.py build")
    p.add_argument("--rag_encoder", type=str, default="biomedclip")
    p.add_argument("--rag_k", type=int, default=5)
    thr = p.add_mutually_exclusive_group()
    thr.add_argument("--tau", type=float, default=None, help="absolute cosine threshold")
    thr.add_argument("--top_frac", type=float, default=None, help=f"keep the top fraction of tiles (default {DEFAULT_TOP_FRAC})")
    p.add_argument("--min_tiles", type=int, default=1)
    p.add_argument("--gemini_model", type=str, default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--cache_path", type=str, default=None, help="Gemini response cache (default <out_dir>/gemini_cache.json)")
    p.add_argument("--region_prompt", type=str, default=DEFAULT_REGION_PROMPT)
    p.add_argument("--task_context", type=str, default=None, help="override the task context L")
    p.add_argument("--text_encoder", type=str, default="conch", help=f"grounding text encoder: {sorted(TEXT_ENCODERS)}")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--conch_checkpoint", type=str, default=None, help="local CONCH weights (else HF hub, token from HF_TOKEN)")
    p.add_argument("--thumbnail_max_side", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--slides", type=str, default=None, help="comma separated slide ids (default: all in the embedding dir)")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.text_encoder not in TEXT_ENCODERS:
        parser.error(f"unknown --text_encoder {args.text_encoder!r}; available: {sorted(TEXT_ENCODERS)}")

    giga_dir = embedding_dir(args.data_path, args.dataset_name, args.embedding_name)
    conch_dir = embedding_dir(args.data_path, args.dataset_name, args.conch_embedding_name)
    out_dir = Path(args.out_dir)
    slides = [s.strip() for s in args.slides.split(",") if s.strip()] if args.slides else list_slides(giga_dir)
    if args.limit is not None:
        slides = slides[: args.limit]
    task_ctx = args.task_context if args.task_context is not None else task_context(args.dataset_name)

    text_encoder = describer = retriever = None
    if args.mode != "all":
        text_encoder = TEXT_ENCODERS[args.text_encoder](device=args.device, checkpoint=args.conch_checkpoint)
        if args.rag_index:
            from tools import slidemix_rag

            if args.rag_encoder not in slidemix_rag.ENCODERS:
                parser.error(f"unknown --rag_encoder {args.rag_encoder!r}; available: {sorted(slidemix_rag.ENCODERS)}")
            retriever = slidemix_rag.Retriever(args.rag_index, slidemix_rag.ENCODERS[args.rag_encoder](device=args.device))
    if args.mode == "gemini":
        describer = GeminiRegionDescriber(model=args.gemini_model, cache_path=args.cache_path or out_dir / "gemini_cache.json")

    results = []
    for i, sid in enumerate(slides, 1):
        giga_path = giga_dir / f"{sid}.safetensors"
        try:
            s = select_roi_for_slide(
                sid,
                args.mode,
                giga_path,
                out_dir,
                conch_path=conch_dir / f"{sid}.safetensors",
                text_encoder=text_encoder,
                describer=describer,
                retriever=retriever,
                wsi_dir=args.wsi_dir,
                task_ctx=task_ctx,
                region_prompt=args.region_prompt,
                rag_k=args.rag_k,
                tau=args.tau,
                top_frac=args.top_frac,
                min_tiles=args.min_tiles,
                overwrite=args.overwrite,
                thumbnail_max_side=args.thumbnail_max_side,
            )
        except Exception as e:
            s = {"slide_id": sid, "mode": args.mode, "effective_mode": None, "status": "error", "fallback": None, "error": f"{type(e).__name__}: {e}"}
        results.append(s)
        detail = s.get("error") or (f"{s['n_selected']}/{s['n_tiles']} tiles" if s.get("n_tiles") is not None else "")
        note = f" [fallback: {s['fallback']}]" if s.get("fallback") else ""
        print(f"[{i}/{len(slides)}] {sid}: {s['status']} {s.get('effective_mode') or ''} {detail}{note}".rstrip(), flush=True)

    counts = {k: sum(1 for r in results if r["status"] == k) for k in ("written", "skipped", "error")}
    counts["fallback"] = sum(1 for r in results if r.get("fallback"))
    summary = {
        "dataset_name": args.dataset_name,
        "mode": args.mode,
        "embedding_dir": str(giga_dir),
        "conch_dir": str(conch_dir),
        "task_context": task_ctx,
        "region_prompt": args.region_prompt,
        "tau": args.tau,
        "top_frac": args.top_frac if args.tau is not None or args.top_frac is not None else DEFAULT_TOP_FRAC,
        "gemini_model": args.gemini_model if args.mode == "gemini" else None,
        "rag_index": args.rag_index,
        "counts": counts,
        "slides": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"out_dir": str(out_dir), "counts": counts}, indent=2))
    return summary


if __name__ == "__main__":
    main()
