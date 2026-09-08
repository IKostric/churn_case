import numpy as np
import pandas as pd
import pytest

from src.data.validate import SNAPSHOT_DATE
from src.features.annotate import build_prompt, parse_label, _resolve_backends
from src.features.clusters import (
    CLUSTER_COLUMN,
    build_cluster_features,
    fit_clusters,
    representative_inquiries,
)
from src.features.embeddings import clean_inquiry_text, embedding_columns

N_DIMENSIONS = 4


def _inquiries(cfg) -> pd.DataFrame:
    """Three well-separated blobs of inquiries across six customers."""
    rng = np.random.default_rng(0)
    centres = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    rows, vectors = [], []
    for i in range(60):
        blob = i % 3
        customer = f"K{i % 6:03d}"
        vectors.append(centres[blob] + rng.normal(0, 0.02, size=N_DIMENSIONS))
        rows.append({
            "henvendelse_id": f"H{i:04d}",
            "kunde_id": customer,
            "dato": SNAPSHOT_DATE - pd.Timedelta(days=i),
            "kanal": "chat",
            "text": f"blob {blob} inquiry {i}",
        })
    frame = pd.DataFrame(rows)
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return pd.concat([frame, pd.DataFrame(matrix, columns=embedding_columns(N_DIMENSIONS))], axis=1)


@pytest.fixture
def cluster_cfg(cfg):
    small = {k: v for k, v in cfg.items()}
    small["clusters"] = {
        **cfg["clusters"],
        "embedding_dim": N_DIMENSIONS,
        "k_candidates": [2, 3],
        "silhouette_sample": 40,
    }
    return small


def test_clustering_recovers_separated_blobs(cluster_cfg):
    inquiries = _inquiries(cluster_cfg)
    train_customers = {"K000", "K001", "K002", "K003"}
    model, assigned, scores = fit_clusters(inquiries, train_customers, cluster_cfg)

    assert model.n_clusters == 3, "silhouette should prefer the true blob count"
    assert set(scores["k"]) == {2, 3}
    # every inquiry gets a cluster, including the held-out customers' ones
    assert assigned[CLUSTER_COLUMN].notna().all()
    assert assigned["distance_to_centroid"].ge(0).all()
    # inquiries from the same blob share a cluster
    blob = assigned["text"].str.split().str[1]
    assert assigned.groupby(blob)[CLUSTER_COLUMN].nunique().eq(1).all()


def test_clustering_is_fitted_on_training_inquiries_only(cluster_cfg, monkeypatch):
    """The KMeans fit must never see a test customer's inquiry."""
    inquiries = _inquiries(cluster_cfg)
    train_customers = {"K000", "K001"}
    seen = {}

    from sklearn.cluster import KMeans

    original_fit = KMeans.fit

    def spy_fit(self, X, *args, **kwargs):
        seen.setdefault("n_rows", X.shape[0])
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(KMeans, "fit", spy_fit)
    fit_clusters(inquiries, train_customers, cluster_cfg)

    n_train_inquiries = int(inquiries["kunde_id"].isin(train_customers).sum())
    assert seen["n_rows"] == n_train_inquiries < len(inquiries)


def test_cluster_features_are_one_share_per_topic(cluster_cfg):
    inquiries = _inquiries(cluster_cfg)
    _, assigned, _ = fit_clusters(inquiries, set(inquiries["kunde_id"]), cluster_cfg)
    customer_ids = pd.Series(sorted(set(inquiries["kunde_id"])) + ["K999"])  # K999 has no inquiries

    features = build_cluster_features(assigned, customer_ids, 3, cluster_cfg).set_index("kunde_id")

    # exactly one share column per topic, and nothing else
    assert set(features.columns) == {f"share_inquiries_cluster_{c:02d}" for c in range(3)}

    shares = features.loc[features.index != "K999"]
    assert shares.sum(axis=1).round(6).eq(1.0).all(), "a customer's shares must sum to 1"

    # no inquiries at all -> zero share everywhere, not NaN
    assert (features.loc["K999"] == 0.0).all()


def test_representatives_are_nearest_the_centroid(cluster_cfg):
    inquiries = _inquiries(cluster_cfg)
    _, assigned, _ = fit_clusters(inquiries, set(inquiries["kunde_id"]), cluster_cfg)
    representatives = representative_inquiries(assigned, n_per_cluster=5)

    assert len(representatives) == 3 * 5
    for cluster, group in representatives.groupby(CLUSTER_COLUMN):
        cutoff = assigned.loc[assigned[CLUSTER_COLUMN] == cluster, "distance_to_centroid"].nsmallest(5).max()
        assert group["distance_to_centroid"].max() <= cutoff + 1e-9


def test_text_cleaning_preserves_language():
    raw = pd.Series(["  Hvorfor  økte\trenten\x07min? ", None])
    cleaned = clean_inquiry_text(raw)
    assert cleaned.iloc[0] == "Hvorfor økte renten min?"  # Norwegian characters intact
    assert cleaned.iloc[1] == ""


def test_annotation_response_parsing():
    assert parse_label('{"name": "Card issues", "description": "Cards fail."}') == {
        "name": "Card issues", "description": "Cards fail."
    }
    # models often wrap JSON in prose or fences
    wrapped = 'Here you go:\n```json\n{"name": "Fees and pricing", "description": "Cost questions."}\n```'
    assert parse_label(wrapped)["name"] == "Fees and pricing"
    assert parse_label("no json here") is None
    assert parse_label('{"description": "missing name"}') is None


def test_reasoning_blocks_do_not_break_parsing():
    """Hybrid-reasoning models (Qwen3) may inline a <think> block whose prose
    contains braces. Ollama splits it into a separate field, but a raw
    transformers run does not - so the parser strips it either way."""
    from src.features.annotate import strip_reasoning

    inlined = (
        '<think>They want JSON like {"name": ...}. Let me think about cards.</think>\n'
        '{"name": "Card issues", "description": "Cards fail."}'
    )
    assert strip_reasoning(inlined).startswith("{")
    assert parse_label(inlined) == {"name": "Card issues", "description": "Cards fail."}

    # a truncated reasoning block (hit the token cap) yields no label, not a crash
    truncated = '<think>Okay, the user wants me to reply with only JSON. The example'
    assert strip_reasoning(truncated) == ""
    assert parse_label(truncated) is None


def test_prompt_includes_every_example():
    prompt = build_prompt(["første henvendelse", "andre henvendelse"])
    assert "første henvendelse" in prompt and "andre henvendelse" in prompt
    assert "JSON" in prompt


def test_labels_are_cached_and_invalidated_by_a_changed_clustering(cfg, tmp_path):
    """Annotation costs ~20 min of local inference, so an unchanged clustering
    must reuse the saved labels - but a different clustering must not."""
    from src.features.annotate import annotate_clusters, load_cached_labels

    scoped = {**cfg, "paths": {**cfg["paths"], "metrics": tmp_path}}
    representatives = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "henvendelse_id": ["H1", "H2", "H3", "H4"],
        "text": ["kort virker ikke", "kortet mitt sluttet", "renten økte", "hvorfor høyere rente"],
    })

    assert load_cached_labels(representatives, scoped) is None  # nothing cached yet

    # tfidf backend keeps this test offline and deterministic
    offline = {**scoped, "clusters": {**scoped["clusters"],
                                      "annotation": {**scoped["clusters"]["annotation"],
                                                     "backend": "tfidf"}}}
    written = annotate_clusters(representatives, offline)
    assert (tmp_path / "cluster_labels.csv").exists()
    assert (tmp_path / "cluster_labels.meta.json").exists()

    cached = load_cached_labels(representatives, offline)
    assert cached is not None
    assert cached["name"].tolist() == written["name"].tolist()

    # a clustering over different inquiries invalidates the cache
    moved = representatives.assign(henvendelse_id=["H9", "H8", "H7", "H6"])
    assert load_cached_labels(moved, offline) is None


def test_backend_resolution_uses_the_endpoint_only_when_configured(cfg):
    annotation = cfg["clusters"]["annotation"]

    def scoped(**overrides):
        return {**cfg, "clusters": {**cfg["clusters"],
                                    "annotation": {**annotation, **overrides}}}

    served = {**annotation["served"], "base_url": "http://localhost:11434/v1"}
    assert _resolve_backends(scoped(backend="auto", served=served)) == ["served"]

    # no endpoint configured, so there is nothing to call and labels fall to tfidf
    assert _resolve_backends(
        scoped(backend="auto", served={**annotation["served"], "base_url": None})
    ) == []
    assert _resolve_backends(scoped(backend="tfidf")) == []
