"""GitHub artifact linkage (docs/DATASET_DEEPENING.md §6; deepening §15-16).

paper -> official repository -> pinned snapshot -> configuration files -> key/value settings -> field-level relation
with the paper's own claims: CONFIRMED_BY_BOTH / CONFLICTING / PRESENT_IN_REPOSITORY / RELATED_BUT_UNCONFIRMED.

Rules:
- A repository is OFFICIAL only with evidence: the paper's own text links it next to a release cue ("our code is
  available at ..."), the PwC archive marks it official, or the arXiv comments field (author-written) links it.
- Repository values are supporting evidence: they never become experiment field values (the repository may differ
  from the exact configuration of the reported experiment). Only artifacts.repository_url is set on experiments.
- The pinned commit is retrieval provenance (what we read, when), not "the commit used in the paper".
- CONFLICTING needs a complete read of the repository's candidate configs (no sampling cut) with one configured value;
  framework / multi-model repositories (LIBRARY_REPOS) only ever yield RELATED_BUT_UNCONFIRMED.
- Access: `git clone --depth 1 --filter=blob:none --no-checkout` pins the commit and lists paths; file bytes come from
  raw.githubusercontent.com at that commit (no GitHub API quota). Files cached gzip-compressed under data/raw/github/
  for re-verification. Only short evidence lines are stored in dataset records.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from services.canonicalization.normalize import NORMALIZATION_VERSION, parse_number
from services.datasets.schemas import (
    Artifact,
    ArtifactEvidence,
    ArtifactRelation,
    ArtifactType,
    EvidenceAssertion,
    EvidenceStatus,
    FailureCategory,
    FieldStatus,
    FieldValue,
    LocationKind,
    Paper,
    ParsedDocument,
    Provenance,
)
from services.datasets.store import DATA, Run, atomic_write_bytes, code_version, now_utc, rel, sha256_bytes, stable_id
from services.evidence.assertions import make
from services.retrieval.http import USER_AGENT

SOURCE = "GITHUB"
VERSION = "artifact-config-0.1.0"
CACHE = DATA / "raw" / "github"
REPO_URL = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)
RELEASE_CUE = re.compile(r"\b(our|the) (code|implementation|source code|models?|data(set)?s?|checkpoints?)\b|\bcode (is|will be|has been)\b|"
                         r"\bavailable (at|on|from)\b|\bwe (release|open-?source|make|provide)\b|\bpublicly available\b|\brepository\b", re.I)
# frameworks / multi-model repositories: a config in them cannot be attributed to one paper's experiment
LIBRARY_REPOS = {"huggingface/transformers", "huggingface/peft", "huggingface/trl", "huggingface/datasets", "huggingface/accelerate",
                 "huggingface/diffusers", "huggingface/pytorch-image-models", "rwightman/pytorch-image-models",
                 "pytorch/pytorch", "pytorch/vision", "pytorch/fairseq", "tensorflow/tensorflow", "tensorflow/models", "tensorflow/tpu",
                 "tensorflow/tensor2tensor", "google-research/google-research", "google-research/big_vision", "google/jax", "google/flax",
                 "keras-team/keras", "facebookresearch/fairseq", "facebookresearch/detectron2", "facebookresearch/maskrcnn-benchmark",
                 "facebookresearch/mmf", "open-mmlab/mmdetection", "open-mmlab/mmsegmentation", "open-mmlab/mmcv", "open-mmlab/mmpretrain",
                 "open-mmlab/mmaction2", "microsoft/deepspeed", "vllm-project/vllm", "nvidia/apex", "nvidia/nemo", "nvidia/megatron-lm",
                 "eleutherai/lm-evaluation-harness", "openai/baselines", "dlr-rm/stable-baselines3", "ultralytics/yolov5",
                 "ultralytics/ultralytics", "lightning-ai/lightning", "pytorchlightning/pytorch-lightning", "allenai/allennlp",
                 "google-research/bert", "openai/human-eval", "stanford-crfm/helm"}
SKIP_PARTS = re.compile(r"(^|/)(node_modules|\.github|docs?|tests?|vendor|third_party|\.vscode|\.devcontainer|assets|images?|figures?|"
                        r"data|datasets|notebooks?|examples/data|site-packages)(/|$)", re.I)
SKIP_NAMES = {"package.json", "package-lock.json", "tsconfig.json", ".pre-commit-config.yaml", "environment.yml", "environment.yaml",
              "codecov.yml", "mkdocs.yml", "readthedocs.yml", ".readthedocs.yaml", "dependabot.yml", "citation.cff"}
CONFIG_EXT = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh")
PATH_CUE = re.compile(r"config|conf|hparam|hyper|exp|recipe|script|train|finetun|ft_|run|launch|args", re.I)
MAX_FILES, MAX_BYTES = 25, 200_000
GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
GIT = ["git", "-c", "credential.helper=", "-c", "core.longpaths=true"]

# leaf key (normalized) -> canonical field. Context-dependent keys are handled in _field_for().
KEYMAP = {
    **dict.fromkeys(["learning_rate", "lr", "base_lr", "max_lr", "peak_lr", "init_lr", "learningrate"], "training.learning_rate"),
    **dict.fromkeys(["batch_size", "train_batch_size", "per_device_train_batch_size", "global_batch_size", "total_batch_size",
                     "train_micro_batch_size_per_gpu", "batchsize"], "training.batch_size"),
    **dict.fromkeys(["num_train_epochs", "epochs", "num_epochs", "max_epochs", "n_epochs", "train_epochs"], "training.epochs"),
    **dict.fromkeys(["max_steps", "num_steps", "train_steps", "total_steps", "num_training_steps", "max_iters", "max_iter", "iterations"], "training.steps"),
    **dict.fromkeys(["warmup_steps", "num_warmup_steps", "warmup_iters", "warmup_updates"], "training.warmup_steps"),
    "warmup_ratio": "training.warmup_ratio",
    **dict.fromkeys(["weight_decay", "wd"], "training.weight_decay"),
    **dict.fromkeys(["dropout", "hidden_dropout_prob", "dropout_rate", "hidden_dropout"], "training.dropout"),
    **dict.fromkeys(["seed", "random_seed"], "training.seed"),
    **dict.fromkeys(["gradient_accumulation_steps", "accumulate_grad_batches", "grad_accum", "gradient_accumulation"], "training.gradient_accumulation"),
    **dict.fromkeys(["max_grad_norm", "gradient_clip_val", "clip_grad_norm", "grad_clip", "clip_grad"], "training.gradient_clipping"),
    **dict.fromkeys(["optimizer", "optim", "optimizer_type", "optimizer_name"], "training.optimizer"),
    **dict.fromkeys(["lr_scheduler_type", "lr_scheduler", "scheduler", "lr_schedule", "scheduler_type"], "training.learning_rate_schedule"),
    **dict.fromkeys(["lora_r", "lora_rank"], "training.peft_rank"),
    "lora_alpha": "training.peft_alpha",
    **dict.fromkeys(["target_modules", "lora_target_modules", "lora_target"], "training.peft_target_modules"),
    **dict.fromkeys(["model_name_or_path", "base_model", "pretrained_model_name_or_path", "base_model_name_or_path"], "model.base_model"),
    "temperature": "inference.temperature",
    "top_p": "inference.top_p",
    **dict.fromkeys(["max_new_tokens", "max_gen_tokens", "max_generation_tokens"], "inference.max_tokens"),
}
BOOL_FIELDS = {"fp16": ("training.precision", "fp16"), "bf16": ("training.precision", "bf16"), "tf32": ("training.precision", "tf32"),
               "load_in_4bit": ("training.quantization", "4-bit"), "load_in_8bit": ("training.quantization", "8-bit")}
LINE_KV = re.compile(r"""^\s*["']?(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*[:=]\s*(?P<val>[^#\n]*?)\s*,?\s*(?:#.*)?$""")
CLI_KV = re.compile(r"--(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)(?:\s+|=)(?P<val>[^\s\\]+)")


@dataclass
class Candidate:
    owner: str
    repo: str
    basis: str  # PAPER_TEXT_URL | PWC_OFFICIAL | ARXIV_COMMENT
    official: bool
    detail: str


@dataclass
class Snapshot:
    repository: str
    commit: str
    files: dict[str, bytes] = field(default_factory=dict)
    license_hint: str | None = None
    complete: bool = False  # every candidate configuration file was fetched (no MAX_FILES cut, no size skip, no failed download)
    retrieved_at: str | None = None  # ISO UTC; set when the snapshot is pinned (store_files) or replayed (load_snapshot)


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


def candidates(paper: Paper, doc: ParsedDocument | None, pwc_links: list[dict], arxiv_comments: str | None) -> list[Candidate]:
    out: dict[str, Candidate] = {}

    def add(owner: str, repo: str, basis: str, detail: str) -> None:
        repo = re.sub(r"\.git$|[).,;:]+$", "", repo)
        key = f"{owner.lower()}/{repo.lower()}"
        if key in LIBRARY_REPOS and basis != "PWC_OFFICIAL":
            return
        out.setdefault(key, Candidate(owner, repo, basis, True, detail))

    if doc is not None:
        for unit_id, text in [(p.paragraph_id, p.text) for p in doc.paragraphs] + [(f.footnote_id, f.text) for f in doc.footnotes]:
            for m in REPO_URL.finditer(text):
                window = text[max(0, m.start() - 200): m.end() + 50]
                if RELEASE_CUE.search(window) or unit_id.startswith("footnote"):
                    add(m.group(1), m.group(2), "PAPER_TEXT_URL", unit_id)
    for link in pwc_links:
        m = REPO_URL.match(link.get("repo_url") or "")
        if m and link.get("is_official"):
            add(m.group(1), m.group(2), "PWC_OFFICIAL", link["repo_url"])
    for m in REPO_URL.finditer(arxiv_comments or ""):
        add(m.group(1), m.group(2), "ARXIV_COMMENT", m.group(0))
    return list(out.values())[:2]


def _git(args: list[str], cwd: str | None = None, timeout: int = 90) -> str:
    p = subprocess.run(GIT + args, cwd=cwd, env=GIT_ENV, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip()[:300])
    return p.stdout


def _rmtree(path: str) -> None:
    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def _select(listing: list[tuple[str, int]]) -> tuple[list[str], bool]:
    """(files to fetch, whether that covers every candidate configuration file)."""
    scored = []
    for path, size in listing:
        name = path.rsplit("/", 1)[-1].lower()
        if path.lower() == "readme.md":
            scored.append((100, path))
            continue
        if not path.lower().endswith(CONFIG_EXT) or name in SKIP_NAMES or size > MAX_BYTES or SKIP_PARTS.search(path):
            continue
        s = (3 if PATH_CUE.search(path) else 0) + (2 if path.lower().endswith((".yaml", ".yml")) else 1) - path.count("/") * 0.25
        scored.append((s, path))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked = [p for _, p in scored[:MAX_FILES]]
    for lic in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE"):
        if any(p == lic for p, _ in listing):
            picked.append(lic)
            break
    return picked, len(scored) <= MAX_FILES


def snapshot(owner: str, repo: str) -> Snapshot:
    """Blobless shallow clone -> pinned commit + selected files (read into memory, temp clone removed)."""
    tmp = tempfile.mkdtemp(prefix="rx_gh_")
    try:
        _git(["clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", f"https://github.com/{owner}/{repo}", "r"], cwd=tmp, timeout=120)
        r = os.path.join(tmp, "r")
        sha = _git(["rev-parse", "HEAD"], cwd=r).strip()
        listing = []
        # no "-l": sizes would make git fetch every missing blob of a blobless clone one by one
        for line in _git(["ls-tree", "-r", "HEAD"], cwd=r, timeout=60).splitlines():
            meta, path = line.split("\t", 1)
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                listing.append((path, 0))
        picked, complete = _select(listing)
        snap = Snapshot(f"{owner}/{repo}", sha)
        # file bytes from raw.githubusercontent.com at the pinned commit (lazy blob fetches of partial clones are unreliable)
        with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            for p in picked:
                resp = client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{quote(p)}")
                if resp.status_code == 200 and len(resp.content) <= MAX_BYTES:  # oversized files are skipped, never truncated
                    snap.files[p] = resp.content
        snap.complete = complete and all(p in snap.files for p in picked)
        lic =next((snap.files[p] for p in snap.files if p.upper().startswith("LICEN")), b"").decode("utf-8", "replace")
        snap.license_hint = ("MIT" if "MIT License" in lic or "Permission is hereby granted" in lic else
                             "Apache-2.0" if "Apache License" in lic else "BSD" if "Redistribution and use in source" in lic else
                             "GPL" if "GNU GENERAL PUBLIC LICENSE" in lic.upper() else "CC" if "Creative Commons" in lic else
                             ("unrecognised" if lic else None))
        return snap
    finally:
        _rmtree(tmp)


def _field_for(path_keys: list[str], leaf: str, raw: str) -> tuple[str, str] | None:
    k = _norm_key(leaf)
    ctx = "/".join(_norm_key(x) for x in path_keys)
    if k in BOOL_FIELDS:
        if raw.strip().strip("\"'").lower() in ("true", "1", "yes"):
            return BOOL_FIELDS[k]
        return None
    if k == "precision" and raw.strip().strip("\"'").lower() in ("16", "bf16", "fp16", "16-mixed", "bf16-mixed", "32", "fp32"):
        return "training.precision", raw.strip().strip("\"'")
    if k == "r" and "lora" in ctx:
        return "training.peft_rank", raw
    if k == "alpha" and "lora" in ctx:
        return "training.peft_alpha", raw
    if k in ("top_k",) and re.search(r"generat|sampl|decod", ctx):
        return "inference.top_k", raw
    if k in KEYMAP:
        return KEYMAP[k], raw
    return None


def parse_values(path: str, text: str) -> list[tuple[str, str, int, str]]:
    """(field, raw value, 1-based line number, verbatim line) for recognised settings in one file."""
    out = []
    stack: list[tuple[int, str]] = []
    is_shell = path.lower().endswith(".sh") or path.lower() == "readme.md"
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 400:
            continue
        if is_shell or "--" in line:
            for m in CLI_KV.finditer(line):
                hit = _field_for([], m.group("key"), m.group("val"))
                if hit and _plausible(hit[1]):
                    out.append((hit[0], hit[1].strip("\"'"), i, line.strip()))
            if is_shell:
                continue
        m = LINE_KV.match(line)
        if not m:
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key, val = m.group("key"), m.group("val").strip()
        if val in ("", "{", "["):
            stack.append((indent, key))
            continue
        hit = _field_for([k for _, k in stack], key.split(".")[-1], val)
        if hit and _plausible(hit[1]):
            out.append((hit[0], hit[1].strip().strip("\"'"), i, line.strip()))
    return out


def _plausible(raw: str) -> bool:
    v = raw.strip().strip("\"',")
    return bool(v) and len(v) <= 80 and not v.startswith(("$", "{", "[", "${", "<")) and v.lower() not in ("none", "null", "~", "auto")


def _norm_value(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().strip("\"'").lower()
    num = parse_number(s.rstrip("%"))
    return repr(round(num[0], 12)) if num else s


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_cache_path(repo_path: str) -> str:
    """Repository path -> path valid on every filesystem (Windows forbids <>:"|?* , control chars, trailing dots/spaces,
    reserved names). Only the cache location changes; Artifact.path keeps the true repository path."""
    segs = []
    for seg in repo_path.split("/"):
        s = "".join(f"%{ord(c):02X}" if c in '<>:"|?*%\\' or ord(c) < 32 else c for c in seg)
        if s.endswith((".", " ")):
            s += "_"
        if s.split(".")[0].upper() in _WIN_RESERVED:
            s = "_" + s
        segs.append(s)
    return "/".join(segs)


def store_files(snap: Snapshot) -> dict[str, tuple[str, str]]:
    """Cache files gzip-compressed plus snapshot.json (the pin); returns path -> (repo-relative cache path, sha256)."""
    out = {}
    base = CACHE / snap.repository.replace("/", "__") / snap.commit
    for p, data in snap.files.items():
        dest = base / f"{safe_cache_path(p)}.gz"
        atomic_write_bytes(dest, gzip.compress(data, mtime=0))
        out[p] = (rel(dest), sha256_bytes(data))
    snap.retrieved_at = snap.retrieved_at or now_utc().isoformat()
    pin = {"repository": snap.repository, "commit": snap.commit, "files": sorted(snap.files), "license_hint": snap.license_hint,
           "complete": snap.complete, "retrieved_at": snap.retrieved_at, "version": VERSION}
    atomic_write_bytes(base / "snapshot.json", json.dumps(pin, indent=1, sort_keys=True).encode())
    return out


def load_snapshot(owner: str, repo: str) -> Snapshot | None:
    """Replay the pinned snapshot of a repository from the cache (latest pin), or None when it was never pinned."""
    pins = [json.loads(p.read_text(encoding="utf-8")) for p in (CACHE / f"{owner}__{repo}").glob("*/snapshot.json")]
    if not pins:
        return None
    pin = max(pins, key=lambda m: m["retrieved_at"])
    base = CACHE / f"{owner}__{repo}" / pin["commit"]
    files = {p: gzip.decompress((base / f"{safe_cache_path(p)}.gz").read_bytes()) for p in pin["files"]}
    return Snapshot(pin["repository"], pin["commit"], files, pin.get("license_hint"), pin.get("complete", False), pin["retrieved_at"])


def link(paper: Paper, exps: list, cand: Candidate, snap: Snapshot, cached: dict[str, tuple[str, str]], evidence: dict[str, dict], run: Run,
         link_evidence_id: str | None) -> tuple[list[Artifact], list[EvidenceAssertion], list[ArtifactEvidence]]:
    repo_id = stable_id("art", "github", snap.repository.lower())
    retrieved = snap.retrieved_at or now_utc()  # the pin time, so replays reproduce identical records
    prov = Provenance(source_name=SOURCE, source_url=f"https://github.com/{snap.repository}", source_identifier=snap.repository,
                      source_version=snap.commit, retrieved_at=retrieved, processing_version=code_version(), extractor_version=VERSION)
    artifacts = [Artifact(artifact_id=repo_id, artifact_type=ArtifactType.CODE_REPOSITORY, url=f"https://github.com/{snap.repository}",
                          repository=snap.repository, commit=snap.commit, linked_paper_ids=[paper.paper_id], link_basis=cand.basis,
                          official=cand.official, license=snap.license_hint, retrieved_at=retrieved, provenance=prov)]
    evs: list[EvidenceAssertion] = []
    values: dict[str, list[tuple[str, EvidenceAssertion]]] = {}
    permissive = snap.license_hint in ("MIT", "Apache-2.0", "BSD")
    for p, data in sorted(snap.files.items()):
        if p.upper().startswith("LICEN"):
            continue
        raw_path, digest = cached[p]
        fid = stable_id("art", "github", snap.repository.lower(), snap.commit, p)
        url = f"https://github.com/{snap.repository}/blob/{snap.commit}/{p}"
        kind = ArtifactType.README if p.lower() == "readme.md" else ArtifactType.SCRIPT if p.lower().endswith(".sh") else ArtifactType.CONFIG_FILE
        artifacts.append(Artifact(artifact_id=fid, artifact_type=kind, url=url, repository=snap.repository, commit=snap.commit, path=p,
                                  parent_artifact_id=repo_id, linked_paper_ids=[paper.paper_id], link_basis=cand.basis, official=cand.official,
                                  license=snap.license_hint, content_sha256=digest, bytes=len(data), raw_path=raw_path, retrieved_at=retrieved,
                                  provenance=prov.model_copy(update={"source_url": url, "raw_path": raw_path})))
        for fname, raw, line_no, line in parse_values(p, data.decode("utf-8", "replace")):
            num = parse_number(raw.rstrip("%"))
            ev = make(source_name=SOURCE, source_url=url, content_hash=digest, source_record_id=fid, paper_id=paper.paper_id, study_id=paper.study_id,
                      experiment_id=None, field_name=fname, extracted_value=raw, normalized_value=num[0] if num else raw,
                      normalization_rule=num[1] if num else None, normalization_version=NORMALIZATION_VERSION if num else None,
                      evidence_text=line, source_location=f"{url}#L{line_no}", location_kind=LocationKind.ARTIFACT_FILE, artifact_id=fid,
                      source_version=snap.commit, retrieval_timestamp=retrieved, extraction_model=VERSION, extraction_run_id=run.run_id,
                      status=EvidenceStatus.MACHINE_EXTRACTED, redistribution_allowed=True if permissive else None)
            evs.append(ev)
            values.setdefault(fname, []).append((_norm_value(ev.normalized_value), ev))

    relations: list[ArtifactEvidence] = []
    multi = snap.repository.lower() in LIBRARY_REPOS

    def file_of(ev: EvidenceAssertion) -> str:
        return ev.source_location.split("#")[0]

    for fname, vals in values.items():
        distinct = {v for v, _ in vals}
        config_vals = [(v, ev) for v, ev in vals if not file_of(ev).lower().endswith("/readme.md")]
        # one relation per distinct paper statement (a study-level setting copied onto many experiments is ONE claim)
        by_claim: dict[tuple[str, tuple[str, ...]], tuple[list[str], Any]] = {}
        for e in exps:
            g, n = fname.split(".", 1)
            fv = getattr(getattr(e, g), n)
            if fv.status is FieldStatus.KNOWN:
                text_ids = tuple(sorted(i for i in fv.evidence_ids if (evidence.get(i) or {}).get("location_kind") in
                                        ("PAPER_SENTENCE", "PAPER_PARAGRAPH", "PAPER_TABLE_CELL", "PAPER_TABLE", "PAPER_CAPTION", "PAPER_FOOTNOTE")))
                if text_ids:
                    by_claim.setdefault((_norm_value(fv.value), text_ids), ([], fv.value))[0].append(e.experiment_id)
        if not by_claim:
            rel_type = ArtifactRelation.PRESENT_IN_REPOSITORY if cand.official and not multi else ArtifactRelation.RELATED_BUT_UNCONFIRMED
            _, ev0 = vals[0]
            relations.append(_relation(paper, None, fname, rel_type, ev0.extracted_value, None, [ev.evidence_id for _, ev in vals], [],
                                       f"value in {'multi-model library ' if multi else ''}repository {snap.repository}@{snap.commit[:8]}; "
                                       "paper states no value", repo_id))
            continue
        holders = {file_of(ev) for _, ev in vals}
        for (pval, pids), (exp_ids, praw) in by_claim.items():
            exp_id = exp_ids[0] if len(exp_ids) == 1 else None
            match = [ev for v, ev in vals if v == pval]
            if multi:
                rel_type, a_ids, why = ArtifactRelation.RELATED_BUT_UNCONFIRMED, [ev.evidence_id for ev in (match or [ev for _, ev in vals])], \
                    f"multi-model library repository: {'a matching' if match else 'no matching'} value, not attributable to this paper's experiment"
            elif match:
                rel_type, a_ids = ArtifactRelation.CONFIRMED_BY_BOTH, [ev.evidence_id for ev in match]
                why = f"repository and paper state the same value ({len({file_of(ev) for ev in match})} of {len(holders)} file(s) holding this field agree)"
            elif (config_vals and len({v for v, _ in config_vals}) == 1 and len(distinct) == 1 and snap.complete):
                rel_type, a_ids, why = ArtifactRelation.CONFLICTING, [ev.evidence_id for _, ev in config_vals], \
                    "the repository's only configured value differs from the paper"
            else:  # README examples, several configs, a sampled (incomplete) or multi-model repo: inconclusive, never a conflict
                basis = "README example" if not config_vals else "several configs" if len(distinct) > 1 else "sampled subset of configs"
                rel_type, a_ids, why = ArtifactRelation.RELATED_BUT_UNCONFIRMED, [ev.evidence_id for _, ev in vals], \
                    f"repository holds {len(distinct)} value(s) ({basis}), none equal to the paper"
            a_val = match[0].extracted_value if match else vals[0][1].extracted_value
            relations.append(_relation(paper, exp_id, fname, rel_type, a_val, praw, a_ids, list(pids), why, repo_id))
    if link_evidence_id:
        for e in exps:
            if e.artifacts.repository_url.status is FieldStatus.UNKNOWN and cand.official:
                e.artifacts.repository_url = FieldValue(status=FieldStatus.KNOWN, raw_value=f"https://github.com/{snap.repository}",
                                                        value=f"https://github.com/{snap.repository}", evidence_ids=[link_evidence_id])
    return artifacts, evs, relations


def _relation(paper: Paper, exp_id: str | None, fname: str, rel_type: ArtifactRelation, a_val: Any, p_val: Any, a_ids: list[str],
              p_ids: list[str], why: str, repo_id: str) -> ArtifactEvidence:
    return ArtifactEvidence(
        artifact_evidence_id=stable_id("artev", repo_id, paper.paper_id, exp_id, fname, rel_type.value, sorted(p_ids)), artifact_id=repo_id, paper_id=paper.paper_id,
        study_id=paper.study_id, experiment_id=exp_id, field_name=fname, relation=rel_type, artifact_value=a_val, paper_value=p_val,
        artifact_evidence_ids=sorted(set(a_ids)), paper_evidence_ids=sorted(set(p_ids)), rationale=why, method_version=VERSION,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=VERSION, parent_record_ids=[repo_id]),
    )


def link_evidence(paper: Paper, doc: ParsedDocument | None, cand: Candidate, run: Run) -> EvidenceAssertion | None:
    """Evidence that the repository is linked from the paper (paragraph/footnote text) - basis for artifacts.repository_url."""
    if cand.basis != "PAPER_TEXT_URL" or doc is None:
        return None
    texts = {p.paragraph_id: p.text for p in doc.paragraphs} | {f.footnote_id: f.text for f in doc.footnotes}
    text = texts.get(cand.detail, "")
    m = re.search(rf"https?://(?:www\.)?github\.com/{re.escape(cand.owner)}/{re.escape(cand.repo)}[^\s)]*", text, re.I)
    if not m:
        return None
    is_fn = cand.detail.startswith("footnote")
    return make(source_name="ARXIV", source_url=doc.source_url, content_hash=doc.source_sha256, source_record_id=doc.source_record_id,
                paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=None, field_name="artifacts.repository_url",
                extracted_value=m.group(0), normalized_value=f"https://github.com/{cand.owner}/{cand.repo}", evidence_text=m.group(0),
                source_location=f"{doc.source_url}#{cand.detail}", paragraph=None if is_fn else cand.detail, footnote=cand.detail if is_fn else None,
                char_start=None if is_fn else m.start(), char_end=None if is_fn else m.end(),
                location_kind=LocationKind.PAPER_FOOTNOTE if is_fn else LocationKind.PAPER_PARAGRAPH, retrieval_timestamp=doc.parsed_at,
                extraction_model=VERSION, extraction_run_id=run.run_id, status=EvidenceStatus.MACHINE_EXTRACTED,
                redistribution_allowed=doc.license.redistribution_allowed)


def fail(run: Run, owner: str, repo: str, err: Exception) -> None:
    run.fail(FailureCategory.retrieval_failure, f"github {owner}/{repo}", "pinned repository snapshot", f"{type(err).__name__}: {str(err)[:200]}", severity="LOW")
