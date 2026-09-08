"""Name each inquiry cluster with an open-weights instruct model.

Two interchangeable backends, both deterministic so the same clusters always
produce the same labels:

* ``served`` - any OpenAI-compatible endpoint: Ollama locally, or vLLM,
  LM Studio, llama.cpp's server.
* ``tfidf`` - deterministic term-frequency fallback, so a missing model or
  unreachable server degrades the labels instead of breaking the pipeline.

Labels are interpretation only: they land in
``outputs/metrics/cluster_labels.csv`` and never feed clustering or training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request

import pandas as pd

from .clusters import CLUSTER_COLUMN, cluster_name
from .embeddings import INQUIRY_ID

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Qwen3 and other hybrid-reasoning models emit a <think> block before the
# answer; it can contain braces, so strip it before hunting for JSON.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
MAX_EXAMPLE_CHARS = 220

PROMPT_TEMPLATE = """You are labelling topics for a Norwegian retail bank's customer-service analytics.

Below are {n} customer inquiries (Norwegian) that were grouped into one cluster by an embedding model.

{examples}

Give this cluster a short English topic name and a one-sentence English description.
The name must be 2-4 words, like "Mortgage refinancing", "Card issues" or "Fees and pricing".

Reply with only JSON, in exactly this form:
{{"name": "...", "description": "..."}}"""

logger = logging.getLogger(__name__)


def _format_examples(texts: list[str]) -> str:
    return "\n".join(f"{i}. {text[:MAX_EXAMPLE_CHARS]}" for i, text in enumerate(texts, start=1))


def build_prompt(texts: list[str]) -> str:
    return PROMPT_TEMPLATE.format(n=len(texts), examples=_format_examples(texts))


def strip_reasoning(raw: str) -> str:
    """Remove a hybrid-reasoning model's <think> block, closed or truncated."""
    return _UNCLOSED_THINK.sub("", _THINK_BLOCK.sub("", raw)).strip()


def parse_label(raw: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    match = _JSON_OBJECT.search(strip_reasoning(raw))
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name, description = parsed.get("name"), parsed.get("description")
    if not isinstance(name, str) or not name.strip():
        return None
    return {
        "name": " ".join(name.split())[:60],
        "description": " ".join(str(description or "").split())[:300],
    }


def _generate_served(prompts: list[str], cfg: dict) -> list[str]:
    """Call an OpenAI-compatible chat endpoint (vLLM, Ollama, LM Studio, ...)."""
    served = cfg["clusters"]["annotation"]["served"]
    base_url = served["base_url"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(served.get("api_key_env") or "", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("annotating via served endpoint %s (model %s)", base_url, served["model"])
    responses = []
    for prompt in prompts:
        body = {
            "model": served["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg["clusters"]["annotation"]["max_new_tokens"],
            "temperature": 0,  # deterministic labels
        }
        # e.g. chat_template_kwargs.enable_thinking=false for Qwen3, whose
        # reasoning block otherwise eats the token budget before the JSON.
        body.update(served.get("extra_body") or {})
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions", data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=served["timeout_seconds"]) as response:
            result = json.load(response)

        choice = result["choices"][0]
        # A reasoning model that thinks past the token cap returns
        # finish_reason=length with empty content - log it rather than let it
        # look like a mysterious parse failure downstream.
        if choice.get("finish_reason") == "length":
            logger.warning(
                "response truncated at max_new_tokens (%d) - reasoning used %d chars; "
                "raise clusters.annotation.max_new_tokens",
                cfg["clusters"]["annotation"]["max_new_tokens"],
                len(choice["message"].get("reasoning") or ""),
            )
        responses.append(choice["message"].get("content") or "")
    return responses


def _tfidf_labels(representatives: pd.DataFrame) -> pd.DataFrame:
    """Deterministic fallback: name a cluster by its most distinctive terms."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = representatives.groupby(CLUSTER_COLUMN)["text"].agg(" ".join)
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for position, cluster in enumerate(documents.index):
        top = matrix[position].toarray().ravel().argsort()[::-1][:4]
        keywords = [terms[i] for i in top]
        rows.append({
            CLUSTER_COLUMN: cluster,
            "name": ", ".join(keywords[:2]),
            "description": f"Top terms: {', '.join(keywords)}",
            "method": "tfidf_fallback",
        })
    return pd.DataFrame(rows)


GENERATORS = {"served": _generate_served}


def _label_paths(cfg: dict):
    directory = cfg["paths"]["metrics"]
    return directory / "cluster_labels.csv", directory / "cluster_labels.meta.json"


def _label_signature(representatives: pd.DataFrame, cfg: dict) -> dict:
    """Identify the clustering the labels describe.

    The representative inquiry ids pin the exact centroids, so any change to
    the embedding model, dimension, k or seed produces a different digest and
    invalidates the cache. Annotation costs ~20 minutes of local inference and
    the labels feed nothing but human interpretation, so re-running it on an
    unchanged clustering is pure waste.
    """
    annotation = cfg["clusters"]["annotation"]
    ids = "|".join(representatives.sort_values(INQUIRY_ID)[INQUIRY_ID].astype(str))
    return {
        "representatives_digest": hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16],
        "n_clusters": int(representatives[CLUSTER_COLUMN].nunique()),
        "n_representatives": int(cfg["clusters"]["n_representatives"]),
        "embedding_model": cfg["clusters"]["embedding_model"],
        "embedding_dim": cfg["clusters"]["embedding_dim"],
        "backend": annotation["backend"],
        "served_model": (annotation.get("served") or {}).get("model"),
    }


def load_cached_labels(representatives: pd.DataFrame, cfg: dict) -> pd.DataFrame | None:
    labels_path, meta_path = _label_paths(cfg)
    if not (labels_path.exists() and meta_path.exists()):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        cached = json.load(fh)
    expected = _label_signature(representatives, cfg)
    if cached != expected:
        changed = [k for k, v in expected.items() if cached.get(k) != v]
        logger.info("cluster label cache stale (%s changed) - re-annotating", changed)
        return None
    logger.info("reusing cached cluster labels from %s", labels_path.name)
    return pd.read_csv(labels_path)


def _resolve_backends(cfg: dict) -> list[str]:
    """Which generators to try, in order."""
    annotation = cfg["clusters"]["annotation"]
    backend = annotation["backend"]
    if backend != "auto":
        return [backend] if backend in GENERATORS else []
    return ["served"] if (annotation.get("served") or {}).get("base_url") else []


def annotate_clusters(
    representatives: pd.DataFrame, cfg: dict, use_cache: bool = True
) -> pd.DataFrame:
    """Label every cluster, writing labels and raw responses to outputs/metrics."""
    if use_cache and (cached := load_cached_labels(representatives, cfg)) is not None:
        return cached

    backend = cfg["clusters"]["annotation"]["backend"]
    grouped = representatives.groupby(CLUSTER_COLUMN)["text"].agg(list)
    prompts = {cluster: build_prompt(texts) for cluster, texts in grouped.items()}

    # Resolve per cluster, not per backend: one cluster whose reasoning ran past
    # the token cap must not discard the other fifteen good labels.
    resolved: dict[int, dict] = {}
    raw_responses: dict[str, dict[int, str]] = {}
    for candidate in _resolve_backends(cfg):
        pending = [cluster for cluster in prompts if cluster not in resolved]
        if not pending:
            break
        try:
            responses = GENERATORS[candidate]([prompts[c] for c in pending], cfg)
        except (OSError, urllib.error.URLError, KeyError, RuntimeError, ValueError) as error:
            logger.warning("%s annotation unavailable (%s)", candidate, error)
            if backend != "auto":
                raise
            continue

        raw_responses[candidate] = dict(zip(pending, responses))
        for cluster, response in raw_responses[candidate].items():
            if (parsed_label := parse_label(response)) is not None:
                resolved[cluster] = {CLUSTER_COLUMN: cluster, "method": f"{candidate}_llm", **parsed_label}
        still_missing = [c for c in pending if c not in resolved]
        if still_missing:
            logger.warning(
                "%s returned unparseable output for cluster(s) %s of %d - trying the next backend",
                candidate, still_missing, len(prompts),
            )

    missing = [cluster for cluster in prompts if cluster not in resolved]
    if missing:
        logger.warning("using term-frequency labels for cluster(s) %s", missing)
        fallback = _tfidf_labels(representatives).set_index(CLUSTER_COLUMN)
        for cluster in missing:
            resolved[cluster] = {CLUSTER_COLUMN: cluster, **fallback.loc[cluster].to_dict()}

    labels = pd.DataFrame([resolved[cluster] for cluster in sorted(resolved)])
    labels["cluster_key"] = labels[CLUSTER_COLUMN].map(cluster_name)

    labels_path, meta_path = _label_paths(cfg)
    labels.to_csv(labels_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(_label_signature(representatives, cfg), fh, indent=2)
    with open(cfg["paths"]["metrics"] / "cluster_annotation_raw.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "backend_requested": backend,
                "backends_tried": _resolve_backends(cfg),
                "methods_used": labels["method"].value_counts().to_dict(),
                "served": cfg["clusters"]["annotation"].get("served"),
                "prompts": {str(k): v for k, v in prompts.items()},
                "responses": {
                    candidate: {str(k): v for k, v in per_cluster.items()}
                    for candidate, per_cluster in raw_responses.items()
                },
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("cluster labels (%s):\n%s", labels["method"].value_counts().to_dict(),
                labels[[CLUSTER_COLUMN, "name", "description"]].to_string(index=False))
    return labels
