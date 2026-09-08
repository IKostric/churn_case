"""Descriptive analysis. Nothing here feeds a model.

One treatment per kind of data: histograms and summary statistics for numeric
features, level counts for categorical ones, a time plot for the monthly
activity panel, and two comparisons for the relationship pairs.

``relasjoner.csv`` also serves one modelling purpose: grouping related
customers onto the same side of the train/test split, so that correlated
household outcomes do not straddle it.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data.load import CUSTOMER_ID, TARGET

# The raw files carry TARGET (`avgang_6mnd`); build_target renames it to this
# in the modelling table, so the two sources use different column names.
MODEL_TARGET = "target"

logger = logging.getLogger(__name__)


# How a missing value in each feature is dealt with. Anything not named here
# either has no missing values or falls to DEFAULT_MISSING.
STRUCTURAL_MISSING = {
    "rente_boliglaan": "structural: no mortgage, so filled 0 at build; har_laan flags it",
    "alder": "implausible ages (<18, >100) set to NaN, then " ,
    "days_since_last_inquiry": "structural: never contacted, so filled 730 days "
                               "(the log window); n_inquiries flags it",
}
DEFAULT_MISSING = "median + indicator (logistic); native NaN split (CatBoost)"


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Range and missingness of each numeric feature, with the rule applied.

    Missingness is worth reporting per feature rather than in aggregate because
    the reasons differ: `rente_boliglaan` is structurally absent for customers
    with no mortgage, while a missing `alder` is genuinely unknown.
    """
    summary = df[columns].describe().T.rename(columns={"50%": "median"})
    missing = df[columns].isna().mean() * 100
    summary.insert(1, "missing_pct", missing)
    summary.insert(2, "missing_handling", [
        "-" if missing[c] == 0 and c not in STRUCTURAL_MISSING
        else STRUCTURAL_MISSING.get(c, "") + (DEFAULT_MISSING if missing[c] > 0 else "")
        for c in columns
    ])
    return summary.reset_index(names="feature")


def categorical_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Level counts per categorical feature, with each level's churn rate."""
    rows = []
    for column in columns:
        grouped = df.groupby(column, dropna=False)[MODEL_TARGET]
        for level, churn in grouped.mean().items():
            rows.append({"feature": column, "level": level,
                         "customers": int(grouped.size()[level]), "churn_rate": churn})
    return pd.DataFrame(rows).sort_values(["feature", "customers"], ascending=[True, False])


def monthly_activity(engasjement: pd.DataFrame) -> pd.DataFrame:
    """Population mean per month - the sequential view of the activity panel."""
    return (
        engasjement.groupby("maaned")[["innlogginger", "korttransaksjoner"]]
        .mean()
        .reset_index()
    )


def plot_numeric(df: pd.DataFrame, columns: list[str], path) -> None:
    columns = sorted(columns)
    rows = -(-len(columns) // 4)
    fig, axes = plt.subplots(rows, 4, figsize=(15, 2.5 * rows))
    for ax, column in zip(axes.flat, columns):
        ax.hist(df[column].dropna(), bins=30, color="#0072B2")
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    _finish(fig, path, "Numeric features")


def plot_monthly(monthly: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(monthly["maaned"], monthly["innlogginger"], color="#0072B2", label="logins")
    ax.plot(monthly["maaned"], monthly["korttransaksjoner"], color="#D55E00",
            label="card transactions")
    ax.set_ylabel("Population mean per customer")
    ax.legend(frameon=False, fontsize=9)
    _finish(fig, path, "Monthly activity")


def _finish(fig, path, title: str) -> None:
    fig.suptitle(title, fontsize=11) if len(fig.axes) > 1 else fig.axes[0].set_title(title)
    for ax in fig.axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)


def relation_type_per_customer(kunder: pd.DataFrame, relasjoner: pd.DataFrame) -> pd.Series:
    """Each customer's relation type, or "none"."""
    both_directions = pd.concat([
        relasjoner.rename(columns={"kunde_id_1": CUSTOMER_ID}),
        relasjoner.rename(columns={"kunde_id_2": CUSTOMER_ID}),
    ])[[CUSTOMER_ID, "relasjon"]]
    per_customer = both_directions.drop_duplicates(CUSTOMER_ID).set_index(CUSTOMER_ID)["relasjon"]
    return kunder[CUSTOMER_ID].map(per_customer).fillna("none")


def cohort_churn(kunder: pd.DataFrame, relasjoner: pd.DataFrame) -> pd.DataFrame:
    """Churn rate for customers with no relation, each relation type, and any."""
    df = pd.DataFrame({
        "cohort": relation_type_per_customer(kunder, relasjoner).to_numpy(),
        "churned": kunder[TARGET].to_numpy(),
    })
    by_type = df.groupby("cohort")["churned"].agg(customers="size", churners="sum")
    any_relation = df[df.cohort != "none"]["churned"].agg(customers="size", churners="sum")
    table = pd.concat([by_type, any_relation.to_frame("any relation").T])
    table["churn_rate"] = table["churners"] / table["customers"]
    return table.reset_index(names="cohort")


def joint_churn(kunder: pd.DataFrame, relasjoner: pd.DataFrame) -> pd.DataFrame:
    """How often both members of a pair churn, against independence."""
    churned = kunder.set_index(CUSTOMER_ID)[TARGET]
    pairs = relasjoner.assign(
        churn_1=relasjoner["kunde_id_1"].map(churned),
        churn_2=relasjoner["kunde_id_2"].map(churned),
    )
    both = (pairs.churn_1 == 1) & (pairs.churn_2 == 1)
    independent = churned.mean() ** 2

    table = pairs.assign(both=both).groupby("relasjon")["both"].agg(pairs="size", both_churn="mean")
    table.loc["all pairs"] = [len(pairs), both.mean()]
    table["if_independent"] = independent
    table["lift"] = table["both_churn"] / independent
    return table.reset_index(names="relation_type")


def plot_comparison(cohorts: pd.DataFrame, joint: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].bar(cohorts["cohort"], cohorts["churn_rate"] * 100, color="#D55E00")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_title("Churn by relation cohort")

    x = range(len(joint))
    axes[1].bar([i - 0.2 for i in x], joint["both_churn"] * 100, 0.4,
                label="observed", color="#D55E00")
    axes[1].bar([i + 0.2 for i in x], joint["if_independent"] * 100, 0.4,
                label="if independent", color="#0072B2")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(joint["relation_type"])
    axes[1].set_ylabel("Both members churn (%)")
    axes[1].set_title("Joint churn within pairs")
    axes[1].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
    _finish(fig, path, "Relations")


def analyse(clean: dict[str, pd.DataFrame], cfg: dict) -> None:
    """Write one table and one figure per kind of feature."""
    from .features.build import feature_spec, load_feature_set

    metrics, figures = cfg["paths"]["metrics"], cfg["paths"]["figures"]
    table = load_feature_set("main", cfg)
    spec = feature_spec(table, "main")

    numeric = numeric_summary(table, spec["numeric"])
    numeric.to_csv(metrics / "feature_numeric.csv", index=False)
    logger.info("numeric features:\n%s", numeric.round(2).to_string(index=False))
    plot_numeric(table, spec["numeric"], figures / "feature_numeric.png")

    counts = categorical_counts(table, spec["categorical"])
    counts.to_csv(metrics / "feature_categorical.csv", index=False)
    logger.info("categorical features:\n%s", counts.round(4).to_string(index=False))

    monthly = monthly_activity(clean["engasjement"])
    monthly.to_csv(metrics / "feature_monthly_activity.csv", index=False)
    plot_monthly(monthly, figures / "feature_monthly_activity.png")

    kunder, relasjoner = clean["kunder"], clean["relasjoner"]
    cohorts = cohort_churn(kunder, relasjoner)
    joint = joint_churn(kunder, relasjoner)
    cohorts.to_csv(metrics / "relation_cohort_churn.csv", index=False)
    joint.to_csv(metrics / "relation_joint_churn.csv", index=False)
    logger.info("churn by relation cohort:\n%s", cohorts.round(4).to_string(index=False))
    logger.info("joint churn within pairs:\n%s", joint.round(4).to_string(index=False))
    plot_comparison(cohorts, joint, figures / "relation_comparison.png")
