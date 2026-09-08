"""Command-line entry point.

    python -m src.pipeline build-features
    python -m src.pipeline train
    python -m src.pipeline evaluate
    python -m src.pipeline run-all

`main` is built up front; the text and topic blocks are appended to it during
the train stage, since the topic block needs the train/test split to fit its
clustering on training customers only.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .analysis import analyse as run_analysis
from .config import load_config, setup_logging
from .data.clean import clean_all
from .data.load import CUSTOMER_ID, load_raw
from .data.validate import validate_raw
from .evaluation.report import evaluate_all
from .features.annotate import annotate_clusters
from .features.build import add_clusters, add_text, build_main, save_feature_set
from .features.clusters import build_cluster_features, fit_clusters, representative_inquiries
from .features.embeddings import embed_inquiries
from .models.split import build_groups, make_split, save_split
from .models.train import experiments, run_experiment

logger = logging.getLogger(__name__)


def prepare_data(cfg: dict) -> dict[str, pd.DataFrame]:
    raw = load_raw(cfg)
    report = validate_raw(raw)
    report.to_csv(cfg["paths"]["metrics"] / "validation_report.csv", index=False)
    return clean_all(raw)


def build_features(cfg: dict) -> pd.DataFrame:
    """Build the `main` table and the reusable grouped train/test split."""
    clean = prepare_data(cfg)
    main = build_main(clean, cfg)
    save_feature_set(main, "main", cfg)

    groups = build_groups(main[CUSTOMER_ID], clean["relasjoner"])
    split = make_split(main[["kunde_id", "target"]], groups, cfg)
    save_split(split, cfg)
    return main


def train(cfg: dict) -> None:
    """Train every configured experiment, building each feature block once."""
    from .features.build import load_feature_set

    planned = experiments(cfg)
    wanted = {feature_set for _, feature_set in planned}

    def run(feature_set: str, table=None) -> None:
        for family, name in [e for e in planned if e[1] == feature_set]:
            run_experiment(family, name, cfg, table=table)

    run("main")
    if wanted == {"main"}:
        return

    clean = prepare_data(cfg)
    main = load_feature_set("main", cfg)
    cluster_features = build_clusters(clean, cfg) if "main_clusters" in wanted else None

    for suffix, build in (("text", lambda: add_text(main, clean["henvendelser"], cfg)),
                          ("clusters", lambda: add_clusters(main, cluster_features, cfg))):
        feature_set = f"main_{suffix}"
        if feature_set not in wanted:
            continue
        table = build()
        save_feature_set(table, feature_set, cfg)
        run(feature_set, table)

    logger.info("trained %d experiments across %d feature sets", len(planned), len(wanted))


def build_clusters(clean: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Embed inquiries, cluster them on training customers, label the clusters,
    and aggregate to per-customer topic features."""
    from .models.split import load_split

    split = load_split(cfg)
    train_customers = set(split.loc[split["split"] == "train", CUSTOMER_ID])

    inquiries = embed_inquiries(clean["henvendelser"], cfg)
    model, assigned, scores = fit_clusters(inquiries, train_customers, cfg)
    scores.to_csv(cfg["paths"]["metrics"] / "cluster_k_selection.csv", index=False)

    representatives = representative_inquiries(assigned, cfg["clusters"]["n_representatives"])
    representatives.to_csv(cfg["paths"]["metrics"] / "cluster_representatives.csv", index=False)
    annotate_clusters(representatives, cfg)

    assigned.drop(columns=[c for c in assigned.columns if c.startswith("e_")]).to_parquet(
        cfg["paths"]["features"] / "inquiry_clusters.parquet", index=False
    )
    return build_cluster_features(
        assigned, split[CUSTOMER_ID], model.n_clusters, cfg
    )


def analyse(cfg: dict) -> None:
    """Descriptive analysis - not part of any model."""
    run_analysis(prepare_data(cfg), cfg)


def evaluate(cfg: dict) -> None:
    evaluate_all(cfg)


def run_all(cfg: dict) -> None:
    build_features(cfg)
    train(cfg)
    evaluate(cfg)
    analyse(cfg)


COMMANDS = {
    "build-features": build_features,
    "train": train,
    "evaluate": evaluate,
    "analyse": analyse,
    "run-all": run_all,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Churn modelling pipeline")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.config)
    logger.info("running %s (seed=%s)", args.command, cfg["seed"])
    COMMANDS[args.command](cfg)


if __name__ == "__main__":
    main()
