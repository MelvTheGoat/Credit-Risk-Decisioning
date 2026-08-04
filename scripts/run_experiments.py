"""Run the full experimental programme and build the deployable artifact.

Produces every table quoted in ``README.md`` under ``reports/tables/``, the
figures under ``reports/figures/``, and a versioned model bundle under
``artifacts/`` that the API and batch scorer serve.

Usage::

    python -m scripts.run_experiments              # full run
    python -m scripts.run_experiments --quick      # smaller book, skip SHAP
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import replace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts import build_manifest, save_bundle  # noqa: E402
from src.calibration import (  # noqa: E402
    CalibratedModel,
    compare_calibrators,
    fit_calibrators,
    reliability_table,
    select_calibrator,
)
from src.config import (  # noqa: E402
    ARTIFACT_DIR,
    DEFAULT_COSTS,
    DEFAULT_SIMULATION,
    DEFAULT_SPLIT,
    FIGURE_DIR,
    SYNTHETIC_FEATURES,
    TABLE_DIR,
    ensure_directories,
)
from src.engine import DecisionEngine  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_models,
    gains_table,
    precision_at_capacity,
    resampling_experiment,
)
from src.explain import ShapExplainer, worked_example  # noqa: E402
from src.fairness import audit_fairness, decompose_disparity, fairness_cost_frontier  # noqa: E402
from src.models import LightGBMModel, build_model_zoo, fit_models, predict_all  # noqa: E402
from src.monitoring import (  # noqa: E402
    monitoring_snapshot,
    vintage_curves,
    vintage_summary,
)
from src.policy import (  # noqa: E402
    approval_frontier,
    compare_policies,
    compare_threshold_rules,
    cost_sensitivity,
)
from src.reject_inference import (  # noqa: E402
    measure_true_uplift,
    run_reject_inference_study,
    run_uplift_sensitivity,
)
from src.simulate import (  # noqa: E402
    describe_simulation,
    mnar_config,
    observed_training_frame,
    simulate_loan_book,
    split_by_vintage,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("experiments")

FEATURES = SYNTHETIC_FEATURES
PRIMARY_MODEL = "lightgbm"


def save_table(frame: pd.DataFrame, name: str) -> None:
    """Write a results table to CSV and echo a preview.

    Args:
        frame: Table to write.
        name: File stem under ``reports/tables/``.
    """
    path = TABLE_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path.name, len(frame))


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_reliability(
    y: np.ndarray, predictions: dict[str, np.ndarray], name: str, title: str
) -> None:
    """Draw a reliability diagram comparing several prediction sets."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for label, p in predictions.items():
        table = reliability_table(y, p, n_bins=10)
        ax.plot(table["mean_predicted"], table["observed_rate"], "o-", markersize=4, label=label)
    limit = max(
        float(max(np.max(p) for p in predictions.values())), float(np.mean(y)) * 2.0
    )
    ax.set_xlim(0, min(limit * 1.05, 1.0))
    ax.set_ylim(0, min(limit * 1.05, 1.0))
    _style(ax, title, "mean predicted PD", "observed default rate")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def scorecard_gap_decomposition(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    reference_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Attribute the scorecard's AUC gap to binning versus additivity.

    Three things separate a WOE scorecard from a gradient booster, and they are
    routinely conflated:

    1. **Coarse binning** — deliberate information loss bought for stability.
    2. **Additivity** — no interactions representable.
    3. **Monotonicity** — each characteristic forced to move risk one way.

    Comparing the scorecard against an L2 logistic regression on the same raw
    features isolates (1), because both are additive. Comparing that logistic
    regression against the booster isolates (2). Refitting the card without the
    monotonic constraint isolates (3).

    Args:
        X_fit: Training features.
        y_fit: Training default flag.
        X_test: Out-of-time features.
        y_test: Out-of-time default flag.
        reference_predictions: Predictions from the fitted zoo, keyed by model.

    Returns:
        One row per variant with out-of-time AUC and what it isolates.
    """
    from sklearn.metrics import roc_auc_score

    from src.scorecard import Scorecard

    numeric, categorical = list(FEATURES.numeric), list(FEATURES.categorical)
    rows = [
        {
            "variant": "scorecard (8 bins, monotonic) [deployed]",
            "roc_auc": float(roc_auc_score(y_test, reference_predictions["scorecard_woe_lr"])),
            "isolates": "the deployed configuration",
        }
    ]

    for bins, monotonic, label, isolates in (
        (8, False, "scorecard (8 bins, unconstrained)", "cost of the monotonicity constraint"),
        (20, True, "scorecard (20 bins, monotonic)", "cost of coarse binning"),
    ):
        card = Scorecard(
            numeric,
            categorical,
            max_bins=bins,
            min_bin_fraction=0.02 if bins > 8 else 0.05,
            enforce_monotonic=monotonic,
        ).fit(X_fit, y_fit)
        rows.append(
            {
                "variant": label,
                "roc_auc": float(roc_auc_score(y_test, card.predict_proba(X_test))),
                "isolates": isolates,
            }
        )

    rows.append(
        {
            "variant": "logistic L2 (continuous, additive)",
            "roc_auc": float(roc_auc_score(y_test, reference_predictions["logistic_l2"])),
            "isolates": "no binning at all, still additive",
        }
    )
    rows.append(
        {
            "variant": "lightgbm (non-linear)",
            "roc_auc": float(roc_auc_score(y_test, reference_predictions["lightgbm"])),
            "isolates": "value of interactions and thresholds",
        }
    )

    frame = pd.DataFrame(rows)
    frame["gap_vs_lightgbm"] = frame["roc_auc"].max() - frame["roc_auc"]
    return frame


def main(quick: bool = False) -> int:
    """Run the experimental programme.

    Args:
        quick: Use a smaller book and skip the expensive SHAP work.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_directories()

    simulation = replace(DEFAULT_SIMULATION, n_applicants=8_000) if quick else DEFAULT_SIMULATION
    split = DEFAULT_SPLIT
    costs = DEFAULT_COSTS
    summary: dict[str, Any] = {}

    # ---------------------------------------------------------------- data --
    logger.info("=== 0. Synthetic loan book ===")
    book = simulate_loan_book(simulation)
    save_table(describe_simulation(book), "00_simulation_summary")

    parts = split_by_vintage(
        book, split.fit_end_vintage, split.calibration_end_vintage, split.test_start_vintage
    )
    fit_frame = observed_training_frame(parts["fit"])
    calibration_frame = observed_training_frame(parts["calibration"])
    test_frame = observed_training_frame(parts["test"])

    X_fit = fit_frame[FEATURES.model_features]
    y_fit = fit_frame["default_observed"].to_numpy()
    X_calibration = calibration_frame[FEATURES.model_features]
    y_calibration = calibration_frame["default_observed"].to_numpy()
    X_test = test_frame[FEATURES.model_features]
    y_test = test_frame["default_observed"].to_numpy()
    exposure_test = test_frame["loan_amount"].to_numpy()

    logger.info(
        "Split (approved only): fit=%d calibration=%d test=%d", len(X_fit), len(X_calibration), len(X_test)
    )
    summary["split"] = {"fit": len(X_fit), "calibration": len(X_calibration), "test": len(X_test)}

    # ------------------------------------------------------- 1-2. models ----
    logger.info("=== 1-2. Scorecard baseline and challenger models ===")
    zoo = fit_models(build_model_zoo(FEATURES), X_fit, y_fit)
    scorecard = zoo["scorecard_woe_lr"].scorecard  # type: ignore[attr-defined]

    save_table(scorecard.encoder.information_values(), "01_information_values")
    save_table(scorecard.points_table(), "01_scorecard_points")
    save_table(scorecard.coefficient_table(), "01_scorecard_coefficients")

    raw_test = predict_all(zoo, X_test)
    save_table(evaluate_models(raw_test, y_test), "02_discrimination_uncalibrated")

    # Decompose the scorecard's gap to the booster. "The scorecard loses" is
    # not a finding until you know WHAT it loses to: the coarse binning it does
    # deliberately for stability, or its inability to represent interactions.
    # The two have completely different remedies.
    save_table(
        scorecard_gap_decomposition(X_fit, y_fit, X_test, y_test, raw_test),
        "02_scorecard_gap_decomposition",
    )

    # -------------------------------------------------------- 3. calibration
    logger.info("=== 3. Calibration ===")
    calibrated: dict[str, np.ndarray] = {}
    chosen_calibrators: dict[str, Any] = {}
    calibration_rows = []

    for name, model in zoo.items():
        fitted = fit_calibrators(model.predict_proba(X_calibration), y_calibration)

        # Out-of-time: real deployment conditions, including drift.
        out_of_time = compare_calibrators(
            fitted, model.predict_proba(X_test), y_test, model_name=name
        )
        out_of_time["evaluation"] = "out_of_time"

        # In-period: isolates calibrator quality from drift, by scoring on the
        # same period the calibrator was fitted on. Optimistic by construction
        # and reported only as a contrast.
        in_period = compare_calibrators(
            fitted, model.predict_proba(X_calibration), y_calibration, model_name=name
        )
        in_period["evaluation"] = "in_period"

        calibration_rows.append(pd.concat([out_of_time, in_period], ignore_index=True))
        selected = select_calibrator(out_of_time)
        chosen_calibrators[name] = fitted[selected]
        calibrated[name] = CalibratedModel(model, fitted[selected], name).predict_proba(X_test)
        logger.info("%s: selected '%s' calibrator", name, selected)

    calibration_comparison = pd.concat(calibration_rows, ignore_index=True)
    save_table(calibration_comparison, "03_calibration_comparison")

    primary_model = zoo[PRIMARY_MODEL]
    primary_fitted = fit_calibrators(primary_model.predict_proba(X_calibration), y_calibration)
    plot_reliability(
        y_test,
        {n: c.transform(primary_model.predict_proba(X_test)) for n, c in primary_fitted.items()},
        "calibration_reliability",
        f"Reliability, {PRIMARY_MODEL}, out-of-time test",
    )

    # ---------------------------------------------------- 4. imbalance ------
    logger.info("=== 4. Imbalance and resampling ===")
    evaluation = evaluate_models(calibrated, y_test)
    save_table(evaluation, "04_discrimination_calibrated")
    save_table(gains_table(y_test, calibrated[PRIMARY_MODEL]), "04_gains_table")
    save_table(precision_at_capacity(y_test, calibrated[PRIMARY_MODEL]), "04_precision_at_capacity")

    resampling = resampling_experiment(
        lambda: LightGBMModel(list(FEATURES.numeric), list(FEATURES.categorical)),
        X_fit,
        y_fit,
        X_test,
        y_test,
    )
    save_table(resampling, "04_resampling")

    # ------------------------------------------------------- 5. policy ------
    logger.info("=== 5. Cost-sensitive decisioning ===")
    policy_comparison = compare_policies(calibrated, y_test, costs=costs, exposure=exposure_test)
    save_table(policy_comparison, "05_policy_cost_comparison")
    save_table(
        pd.concat(
            [
                compare_threshold_rules(y_test, p, costs=costs, exposure=exposure_test, label=n)
                for n, p in calibrated.items()
            ],
            ignore_index=True,
        ),
        "05_threshold_rules",
    )
    frontier = approval_frontier(calibrated, y_test, costs=costs, exposure=exposure_test)
    save_table(frontier, "05_approval_frontier")
    save_table(
        cost_sensitivity(y_test, calibrated[PRIMARY_MODEL], exposure=exposure_test),
        "05_cost_sensitivity",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in calibrated:
        subset = frontier[frontier["model"] == name].sort_values("approval_rate")
        axes[0].plot(subset["approval_rate"], subset["net_cost_per_application"], label=name, linewidth=1.6)
        axes[1].plot(subset["approval_rate"], subset["bad_rate_approved"], label=name, linewidth=1.6)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle=":")
    _style(axes[0], "Cost frontier (lower is better)", "approval rate", "net cost per application")
    _style(axes[1], "Risk of the approved book", "approval rate", "default rate among approved")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cost_frontier.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------- 6. reject inference ------
    logger.info("=== 6. Reject inference (both selection regimes) ===")

    def factory() -> LightGBMModel:
        return LightGBMModel(list(FEATURES.numeric), list(FEATURES.categorical))

    reject_frames = []
    uplift_frames = []
    for regime, configuration in (("MAR", simulation), ("MNAR", mnar_config(simulation))):
        regime_book = simulate_loan_book(configuration)
        regime_parts = split_by_vintage(
            regime_book,
            split.fit_end_vintage,
            split.calibration_end_vintage,
            split.test_start_vintage,
        )
        regime_train = pd.concat(
            [regime_parts["fit"], regime_parts["calibration"]], ignore_index=True
        )
        study = run_reject_inference_study(
            factory,
            regime_train,
            regime_parts["test"],
            FEATURES.model_features,
            list(FEATURES.numeric),
            ["branch_capacity_index"],
        )
        study.insert(0, "regime", regime)
        reject_frames.append(study)

        sensitivity = run_uplift_sensitivity(
            factory, regime_train, regime_parts["test"], FEATURES.model_features
        )
        sensitivity.insert(0, "regime", regime)
        uplift_frames.append(sensitivity)

        baseline = factory().fit(
            regime_train.loc[regime_train["approved"] == 1, FEATURES.model_features],
            regime_train.loc[regime_train["approved"] == 1, "default_observed"].to_numpy(),
        )
        uplift = measure_true_uplift(
            baseline.predict_proba(regime_train[FEATURES.model_features]),
            regime_train["approved"].to_numpy(),
            regime_train["default_true"].to_numpy(),
        )
        uplift.insert(0, "regime", regime)
        uplift_frames.append(uplift.assign(method="measured_truth", uplift_assumed=np.nan))

    save_table(pd.concat(reject_frames, ignore_index=True), "06_reject_inference")
    save_table(pd.concat(uplift_frames, ignore_index=True), "06_uplift_sensitivity")

    # ------------------------------------------------------ 7. fairness -----
    logger.info("=== 7. Fairness audit ===")
    primary_predictions = calibrated[PRIMARY_MODEL]
    per_group, fairness_summary = audit_fairness(
        test_frame,
        y_test,
        primary_predictions,
        list(FEATURES.protected),
        costs.analytic_threshold,
    )
    save_table(per_group, "07_fairness_by_group")
    save_table(fairness_summary, "07_fairness_summary")
    save_table(decompose_disparity(test_frame, primary_predictions, "group"), "07_disparity_decomposition")

    tradeoff = fairness_cost_frontier(
        y_test, primary_predictions, test_frame["group"], costs.analytic_threshold, costs=costs,
        exposure=exposure_test,
    )
    save_table(tradeoff.drop(columns=["thresholds"]), "07_fairness_tradeoff")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(
        tradeoff["demographic_parity_difference"],
        tradeoff["net_cost"] / len(y_test),
        "o-",
        markersize=4,
        label="net cost",
    )
    for _, row in tradeoff.iloc[::4].iterrows():
        ax.annotate(
            f"λ={row['lambda']:.1f}",
            (row["demographic_parity_difference"], row["net_cost"] / len(y_test)),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    _style(
        ax,
        "Cost of approval-rate parity (curve is flat: parity is ~free here)",
        "demographic parity difference",
        "net cost per application",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fairness_tradeoff.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------- 8. explainability ---
    logger.info("=== 8. Explainability ===")
    explainer = None
    if not quick:
        explainer = ShapExplainer(primary_model, X_fit)
        save_table(explainer.global_importance(X_test.head(1500)), "08_shap_global_importance")

    example = worked_example(
        test_frame,
        primary_predictions,
        costs.analytic_threshold,
        scorecard=scorecard,
        explainer=explainer,
        feature_columns=FEATURES.model_features,
        model_version="experiment-run",
    )
    (TABLE_DIR / "08_worked_example.json").write_text(json.dumps(example, indent=2, default=str))
    (TABLE_DIR / "08_adverse_action_notice.txt").write_text(example["adverse_action_notice"])
    logger.info("Worked example: application %s declined at PD %.4f",
                example["application_id"], example["probability_of_default"])

    # ----------------------------------------------------- 9. monitoring ----
    logger.info("=== 9. Monitoring ===")
    reference_scores = CalibratedModel(
        primary_model, chosen_calibrators[PRIMARY_MODEL]
    ).predict_proba(X_fit)
    snapshot = monitoring_snapshot(
        fit_frame,
        test_frame,
        FEATURES.model_features,
        reference_scores,
        primary_predictions,
        y_test,
    )
    save_table(snapshot["feature_drift"], "09_feature_drift")  # type: ignore[arg-type]
    save_table(snapshot["score_distribution"], "09_score_distribution")  # type: ignore[arg-type]
    save_table(snapshot["triggers"], "09_triggers")  # type: ignore[arg-type]

    curves = vintage_curves(book)
    save_table(curves, "09_vintage_curves")
    save_table(vintage_summary(curves), "09_vintage_summary")

    fig, ax = plt.subplots(figsize=(7, 5))
    for cohort, subset in curves.groupby("cohort"):
        ax.plot(subset["months_on_book"], subset["cumulative_default_rate"], marker="o",
                markersize=3, label=str(cohort), linewidth=1.5)
    _style(ax, "Vintage curves by origination cohort", "months on book", "cumulative default rate")
    ax.legend(fontsize=8, frameon=False, title="cohort", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vintage_curves.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------- artifact build -----
    logger.info("=== Building deployable artifact ===")
    best = policy_comparison.iloc[0]
    calibration_row = calibration_comparison[
        (calibration_comparison["model"] == PRIMARY_MODEL)
        & (calibration_comparison["evaluation"] == "out_of_time")
    ].sort_values("brier").iloc[0]
    performance_row = evaluation[evaluation["label"] == PRIMARY_MODEL].iloc[0]

    manifest = build_manifest(
        model_name=PRIMARY_MODEL,
        calibrator_name=str(chosen_calibrators[PRIMARY_MODEL].name),
        feature_names=FEATURES.model_features,
        threshold=costs.analytic_threshold,
        costs=costs,
        calibration_metrics={
            "brier": float(calibration_row["brier"]),
            "ece_quantile": float(calibration_row["ece_quantile"]),
            "calibration_slope": float(calibration_row["calibration_slope"]),
            "calibration_intercept": float(calibration_row["calibration_intercept"]),
        },
        performance_metrics={
            "roc_auc": float(performance_row["roc_auc"]),
            "pr_auc": float(performance_row["pr_auc"]),
            "ks": float(performance_row["ks"]),
            "gini": float(performance_row["gini"]),
        },
        training_rows=len(X_fit),
        training_period=f"vintages 0-{split.fit_end_vintage - 1} (approved applicants only)",
        notes=(
            "Trained on synthetic data. Calibrated on vintages "
            f"{split.fit_end_vintage}-{split.calibration_end_vintage - 1}, tested out of time on "
            f"vintages {split.test_start_vintage}+. Not fit for real lending decisions."
        ),
    )
    version_directory = save_bundle(
        primary_model, chosen_calibrators[PRIMARY_MODEL], manifest, scorecard=scorecard
    )
    np.save(version_directory / "reference_scores.npy", reference_scores)

    # Prove the served artifact reproduces the experiment's own numbers.
    engine = DecisionEngine.from_artifacts()
    served = engine.predict_proba(X_test)
    max_difference = float(np.max(np.abs(served - primary_predictions)))
    logger.info("Served vs experiment max |difference|: %.3e", max_difference)
    if max_difference > 1e-9:
        logger.error("Served artifact does not reproduce experiment predictions")
        return 1

    summary["headline"] = {
        "best_model": str(best["label"]),
        "best_net_cost": float(best["net_cost"]),
        "scorecard_net_cost": float(
            policy_comparison[policy_comparison["label"] == "scorecard_woe_lr"]["net_cost"].iloc[0]
        ),
        "artifact_version": manifest.version,
        "score_psi": float(snapshot["score_psi"]),  # type: ignore[arg-type]
    }
    (TABLE_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info("=== Done. Tables in %s, figures in %s, artifact %s ===",
                TABLE_DIR, FIGURE_DIR, ARTIFACT_DIR / manifest.version)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller book, skip SHAP")
    arguments = parser.parse_args()
    raise SystemExit(main(quick=arguments.quick))
