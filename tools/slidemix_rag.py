import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
EFETCH_BATCH = 200
INTERVAL_NO_KEY = 0.34
INTERVAL_WITH_KEY = 0.11
NCBI_TOOL = "SlideMix"
USER_AGENT = f"{NCBI_TOOL}/1.0 (python urllib)"

BIOMEDCLIP_HF_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

DATASET_CONFIG_DIR = Path(__file__).resolve().parents[1] / "dataset_configs"


def default_http_get(url: str, timeout: float = 30.0, retries: int = 3, backoff: float = 1.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    retries = max(1, int(retries))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise last_err


def _eutils_params(api_key=None, email=None) -> dict:
    params = {"tool": NCBI_TOOL}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def esearch_url(query: str, retmax: int, api_key=None, email=None) -> str:
    params = {"db": "pubmed", "term": query, "retmax": int(retmax), "retmode": "json", "sort": "relevance"}
    params.update(_eutils_params(api_key, email))
    return f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"


def efetch_url(pmids: list, api_key=None, email=None) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    params.update(_eutils_params(api_key, email))
    return f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_efetch_xml(xml) -> list:
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    records = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid = _text(cit.find("PMID"))
        article = cit.find("Article")
        if not pmid or article is None:
            continue
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            txt = _text(ab)
            if not txt:
                continue
            label = ab.get("Label")
            parts.append(f"{label}: {txt}" if label else txt)
        year = _text(article.find("Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline = _text(article.find("Journal/JournalIssue/PubDate/MedlineDate"))
            year = medline[:4] if medline[:4].isdigit() else ""
        records.append(
            {
                "pmid": pmid,
                "title": _text(article.find("ArticleTitle")),
                "abstract": " ".join(parts),
                "journal": _text(article.find("Journal/Title")),
                "year": year,
            }
        )
    return records


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_pubmed(
    queries: list,
    retmax_per_query: int,
    out_jsonl,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    sleep: Optional[float] = None,
    http_get: Optional[Callable[[str], bytes]] = None,
) -> list:
    http_get = http_get or default_http_get
    interval = (INTERVAL_WITH_KEY if api_key else INTERVAL_NO_KEY) if sleep is None else float(sleep)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = {str(d["pmid"]) for d in read_jsonl(out_jsonl)}

    def get(url):
        data = http_get(url)
        if interval > 0:
            time.sleep(interval)
        return data

    new_records = []
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for query in queries:
            res = json.loads(get(esearch_url(query, retmax_per_query, api_key, email)))
            ids = [str(i) for i in res.get("esearchresult", {}).get("idlist", [])][: int(retmax_per_query)]
            todo = list(dict.fromkeys(i for i in ids if i not in seen))
            for start in range(0, len(todo), EFETCH_BATCH):
                batch = todo[start : start + EFETCH_BATCH]
                for rec in parse_efetch_xml(get(efetch_url(batch, api_key, email))):
                    if rec["pmid"] in seen:
                        continue
                    seen.add(rec["pmid"])
                    rec["query"] = query
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    new_records.append(rec)
            f.flush()
    return new_records


def _is_search_term(label: str) -> bool:
    return any(ch.isalpha() for ch in label) and '"' not in label


def default_queries(organ: str, task_names: Iterable, label_names=None) -> list:
    base = f'"{organ}"[Title/Abstract] AND (histopathology OR "whole slide image")'
    task_names = list(task_names or [])
    queries = []
    for task in task_names:
        terms = [str(task).replace("_", " ")]
        if isinstance(label_names, dict):
            labels = label_names.get(task, [])
        else:
            labels = label_names or []
        terms += [str(lab) for lab in labels if _is_search_term(str(lab))]
        terms = [t for t in dict.fromkeys(terms) if t and '"' not in t]
        if terms:
            queries.append(f"{base} AND (" + " OR ".join(f'"{t}"' for t in terms) + ")")
    queries.append(base)
    return list(dict.fromkeys(queries))


def dataset_query_info(dataset_name: str):
    import yaml

    from utils.CONSTS import DATASETS_ORGAN

    organ = DATASETS_ORGAN[dataset_name]
    with open(DATASET_CONFIG_DIR / dataset_name / "task_configs.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    task_types = cfg.get("all_task_dict", {})
    tasks = [t for t in cfg.get("tasks_to_run", list(task_types)) if task_types.get(t, "cls") == "cls"]
    labels = {t: [str(k) for k in cfg.get("label_dict", {}).get(t, {})] for t in tasks}
    return organ, tasks, labels


class TextEncoder(Protocol):
    def encode(self, texts: list) -> np.ndarray:
        pass


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _load_biomedclip(model_name: str, device: str):
    from models.clip import create_model_from_pretrained, get_tokenizer

    model, _preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer


class BiomedCLIPEncoder:

    def __init__(self, device: str = "cpu", batch_size: int = 64, model_name: str = BIOMEDCLIP_HF_ID, loader=None):
        self.device = device
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.name = model_name
        self._loader = loader or _load_biomedclip
        self._model = None
        self._tokenizer = None
        self.dim = None

    @property
    def context_length(self):
        self._ensure_loaded()
        return getattr(self._tokenizer, "context_length", None)

    def _ensure_loaded(self):
        if self._model is None:
            model, tokenizer = self._loader(self.model_name, self.device)
            self._model = model.eval().to(self.device)
            self._tokenizer = tokenizer

    def encode(self, texts: list) -> np.ndarray:
        import torch

        self._ensure_loaded()
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self._tokenizer(list(texts[start : start + self.batch_size])).to(self.device)
                emb = self._model.encode_text(tokens)
                out.append(emb.float().cpu().numpy())
        arr = _l2_normalise(np.concatenate(out)) if out else np.zeros((0, self.dim or 0), np.float32)
        if len(arr):
            self.dim = int(arr.shape[1])
        return arr


ENCODERS = {"biomedclip": BiomedCLIPEncoder}


def _try_import_faiss():
    try:
        import faiss
    except ImportError:
        return None
    return faiss


def doc_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip().rstrip(".")
    abstract = (doc.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def build_index(jsonl, out_dir, encoder, batch_size: int = 256) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs, seen = [], set()
    for d in read_jsonl(jsonl):
        pmid = str(d.get("pmid", ""))
        if not pmid or pmid in seen or not doc_text(d):
            continue
        seen.add(pmid)
        docs.append(d)
    if not docs:
        raise ValueError(f"no documents with text in {jsonl}")

    chunks = [encoder.encode([doc_text(d) for d in docs[s : s + batch_size]]) for s in range(0, len(docs), batch_size)]
    emb = _l2_normalise(np.concatenate(chunks))

    with open(out_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", emb)

    faiss = _try_import_faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out_dir / "index.faiss"))
    elif (out_dir / "index.faiss").exists():
        (out_dir / "index.faiss").unlink()

    meta = {
        "encoder": getattr(encoder, "name", type(encoder).__name__),
        "dim": int(emb.shape[1]),
        "n": int(emb.shape[0]),
        "faiss": faiss is not None,
        "source": str(jsonl),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def format_context(docs: list, max_chars: int = 1500, snippet_chars: int = 200) -> str:
    lines, total = [], 0
    for d in docs:
        title = (d.get("title") or "").strip()
        abstract = " ".join((d.get("abstract") or "").split())
        snippet = abstract[:snippet_chars].rstrip() + ("..." if len(abstract) > snippet_chars else "")
        line = f"- {title}" + (f": {snippet}" if snippet else "")
        sep = 1 if lines else 0
        if total + sep + len(line) > max_chars:
            room = max_chars - total - sep
            if room > 0 and not lines:
                lines.append(line[:room])
            break
        lines.append(line)
        total += sep + len(line)
    return "\n".join(lines)


class Retriever:

    def __init__(self, index_dir, encoder, use_faiss: Optional[bool] = None):
        self.index_dir = Path(index_dir)
        self.encoder = encoder
        with open(self.index_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.docs = read_jsonl(self.index_dir / "docs.jsonl")
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype(np.float32)
        self.index = None
        faiss_path = self.index_dir / "index.faiss"
        if use_faiss is not False and faiss_path.exists():
            faiss = _try_import_faiss()
            if faiss is not None:
                self.index = faiss.read_index(str(faiss_path))
            elif use_faiss:
                raise ImportError("faiss requested but not importable")
        self.backend = "faiss" if self.index is not None else "numpy"

    @staticmethod
    def query_text(L: str, P: str) -> str:
        return f"{L}; {P}"

    def query(self, L: str, P: str, k: int = 5) -> list:
        e_q = _l2_normalise(self.encoder.encode([self.query_text(L, P)]))
        if e_q.shape[1] != self.embeddings.shape[1]:
            raise ValueError(f"query dim {e_q.shape[1]} != index dim {self.embeddings.shape[1]} (encoder mismatch)")
        k = max(0, min(int(k), len(self.docs)))
        if k == 0:
            return []
        if self.index is not None:
            scores, idx = self.index.search(e_q, k)
            pairs = [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        else:
            sims = self.embeddings @ e_q[0]
            top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
            top = top[np.argsort(-sims[top], kind="stable")]
            pairs = [(int(i), float(sims[i])) for i in top]
        out = []
        for i, s in pairs:
            d = self.docs[i]
            out.append({"pmid": str(d.get("pmid")), "title": d.get("title", ""), "abstract": d.get("abstract", ""), "score": s})
        return out

    def format_context(self, docs: list, max_chars: int = 1500, snippet_chars: int = 200) -> str:
        return format_context(docs, max_chars=max_chars, snippet_chars=snippet_chars)


def _make_encoder(name: str, device: str, batch_size: int, parser):
    if name not in ENCODERS:
        parser.error(f"unknown --encoder {name!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[name](device=device, batch_size=batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SlideMix PubMed RAG (fetch / build / query)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download PubMed titles/abstracts via NCBI E-utilities")
    p.add_argument("--out", type=str, required=True, help="output jsonl (appended, deduplicated by PMID)")
    p.add_argument("--queries", type=str, nargs="*", default=None, help="explicit PubMed queries")
    p.add_argument("--dataset_name", type=str, default=None, help="derive organ/tasks/labels from repo configs")
    p.add_argument("--organ", type=str, default=None, help="override organ for generated queries")
    p.add_argument("--retmax", type=int, default=500, help="max PMIDs per query")
    p.add_argument("--email", type=str, default=None)
    p.add_argument("--api_key_env", type=str, default="NCBI_API_KEY", help="env var holding the NCBI api key")
    p.add_argument("--sleep", type=float, default=None, help="override inter-request interval (s)")

    p = sub.add_parser("build", help="encode docs and build the retrieval index")
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--encoder", type=str, default="biomedclip")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=64)

    p = sub.add_parser("query", help="retrieve top-k docs for task context L and region prompt P")
    p.add_argument("--index_dir", type=str, required=True)
    p.add_argument("--L", type=str, required=True, help="task context (organ, staining, task)")
    p.add_argument("--P", type=str, required=True, help="region prompt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--encoder", type=str, default="biomedclip")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--json", action="store_true", help="print results as JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        queries = list(args.queries or [])
        if args.dataset_name:
            organ, tasks, labels = dataset_query_info(args.dataset_name)
            queries += default_queries(args.organ or organ, tasks, labels)
        elif args.organ:
            queries += default_queries(args.organ, [], None)
        if not queries:
            parser.error("fetch needs --queries, --dataset_name or --organ")
        api_key = os.environ.get(args.api_key_env) or None
        new = fetch_pubmed(
            queries, args.retmax, args.out, api_key=api_key, email=args.email, sleep=args.sleep, http_get=default_http_get
        )
        print(f"{len(new)} new documents from {len(queries)} queries -> {args.out}")

    elif args.command == "build":
        encoder = _make_encoder(args.encoder, args.device, args.batch_size, parser)
        meta = build_index(args.docs, args.out_dir, encoder, batch_size=args.batch_size)
        print(json.dumps(meta, indent=2))

    elif args.command == "query":
        encoder = _make_encoder(args.encoder, args.device, args.batch_size, parser)
        results = Retriever(args.index_dir, encoder).query(args.L, args.P, k=args.k)
        if args.json:
            print(json.dumps(results, ensure_ascii=False))
        else:
            for rank, d in enumerate(results, 1):
                print(f"{rank:2d}. [{d['score']:.4f}] PMID {d['pmid']}: {d['title']}")


if __name__ == "__main__":
    main()
