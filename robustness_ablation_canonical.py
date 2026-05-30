"""
Additional robustness and ablation checks for the canonical ABM.

Outputs:
- robustness_ablation_canonical.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore

from abm_simulation_canonical import GOVERNANCE, PARAMS, aggregate


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    skill = result["skills_mean"][-1]
    quality = result["output_quality_mean"][-1]
    return {
        "tipping_point_epoch": result["tipping_point"]["first_crossing_epoch"],
        "final_skill": round(skill, 4),
        "final_quality": round(quality, 4),
        "quality_skill_gap": round(quality - skill, 4),
    }


def run_variant(
    params_update: dict[str, Any] | None = None,
    governance_update: dict[str, Any] | None = None,
    regime_key: str = "loose",
    seed: int = 42,
) -> dict[str, Any]:
    params = PARAMS.copy()
    params["n_monte_carlo"] = 200
    if params_update:
        params.update(params_update)

    regime = GOVERNANCE[regime_key].copy()
    if governance_update:
        regime.update(governance_update)

    return summarize(aggregate(regime_key, regime, params, seed))


def run_mechanism_ablation(seed: int = 42) -> list[dict[str, Any]]:
    variants = [
        ("Canonical", {}),
        ("No Red Queen", {"red_queen_rate": 0.0}),
        ("No lazy contagion", {"lazy_contagion_rate": 0.0}),
        ("No adaptive engagement", {"adaptive_engagement_rate": 0.0, "adaptive_engagement_cap": 0.0}),
        ("No knowledge spillover", {"knowledge_spillover_rate": 0.0}),
    ]
    rows: list[dict[str, Any]] = []
    for label, update in variants:
        hotl = run_variant(update, regime_key="loose", seed=seed)
        hitl = run_variant(update, regime_key="strict", seed=seed)
        rows.append(
            {
                "variant": label,
                "hotl_tipping_point_epoch": hotl["tipping_point_epoch"],
                "hitl_tipping_point_epoch": hitl["tipping_point_epoch"],
                "hotl_final_skill": hotl["final_skill"],
                "hotl_quality_skill_gap": hotl["quality_skill_gap"],
            }
        )
    return rows


def run_red_queen_sweep(seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in [0.0, 0.0025, 0.005, 0.0075, 0.01]:
        update = {"red_queen_rate": value}
        hotl = run_variant(update, regime_key="loose", seed=seed)
        hitl = run_variant(update, regime_key="strict", seed=seed)
        rows.append(
            {
                "red_queen_rate": value,
                "hotl_tipping_point_epoch": hotl["tipping_point_epoch"],
                "hitl_tipping_point_epoch": hitl["tipping_point_epoch"],
                "hotl_final_skill": hotl["final_skill"],
            }
        )
    return rows


def run_structural_sensitivity(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "lazy_contagion_rate": [],
        "adaptive_engagement_cap": [],
        "ai_interaction_floor": [],
        "ai_interaction_form": [],
        "recovery_switch_epoch": [],
    }

    for value in [0.0, 0.15, 0.3, 0.45]:
        row = run_variant({"lazy_contagion_rate": value}, regime_key="loose", seed=seed)
        row["param_value"] = value
        out["lazy_contagion_rate"].append(row)

    for value in [0.0, 0.15, 0.3, 0.45]:
        row = run_variant({"adaptive_engagement_cap": value}, regime_key="loose", seed=seed)
        row["param_value"] = value
        out["adaptive_engagement_cap"].append(row)

    for value in [0.25, 0.5, 0.75]:
        row = run_variant({"ai_interaction_floor": value}, regime_key="loose", seed=seed)
        row["param_value"] = value
        out["ai_interaction_floor"].append(row)

    for form in ["linear", "concave", "convex", "threshold"]:
        row = run_variant({"ai_interaction_form": form}, regime_key="loose", seed=seed)
        row["param_value"] = form
        out["ai_interaction_form"].append(row)

    for switch_epoch in [10, 18, 30, 45]:
        row = run_variant({}, {"switch_epoch": switch_epoch}, regime_key="recovery", seed=seed)
        row["param_value"] = switch_epoch
        out["recovery_switch_epoch"].append(row)

    return out


def run_joint_stress_tests(seed: int = 42) -> list[dict[str, Any]]:
    """Stress-test selected interacting assumptions rather than one parameter at a time."""
    scenarios: list[tuple[str, str, dict[str, Any], dict[str, Any], str]] = [
        (
            "Canonical HOTL",
            "loose",
            {},
            {},
            "Baseline structural scenario.",
        ),
        (
            "Favorable HOTL boundary",
            "loose",
            {"red_queen_rate": 0.0025, "alpha_atrophy": 0.005, "ai_interaction_floor": 0.75},
            {"base_engagement": 0.5},
            "Slow threshold growth, low atrophy, higher passive engagement, and easier AI extraction.",
        ),
        (
            "Intermediate HOTL boundary",
            "loose",
            {"red_queen_rate": 0.0025, "alpha_atrophy": 0.005, "ai_interaction_floor": 0.5},
            {"base_engagement": 0.3},
            "Lower atrophy and slower threshold growth with moderate passive engagement.",
        ),
        (
            "Adverse HOTL stress",
            "loose",
            {"red_queen_rate": 0.01, "alpha_atrophy": 0.02, "ai_interaction_floor": 0.25},
            {"base_engagement": 0.1},
            "Fast threshold growth, high atrophy, low passive engagement, and harder AI extraction.",
        ),
        (
            "Canonical HITL",
            "strict",
            {},
            {},
            "Baseline strict-verification scenario.",
        ),
        (
            "Favorable HITL boundary",
            "strict",
            {"red_queen_rate": 0.0, "alpha_atrophy": 0.005},
            {"base_engagement": 0.9, "penalty": 0.05},
            "No Red Queen pressure, low atrophy, high verification engagement, and lower penalty.",
        ),
        (
            "Adverse HITL stress",
            "strict",
            {"red_queen_rate": 0.01, "alpha_atrophy": 0.02},
            {"base_engagement": 0.5, "penalty": 0.25},
            "Fast threshold growth, high atrophy, lower verification engagement, and higher penalty.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, regime_key, params_update, governance_update, interpretation in scenarios:
        row = run_variant(params_update, governance_update, regime_key=regime_key, seed=seed)
        row["scenario"] = label
        row["regime"] = regime_key
        row["interpretation"] = interpretation
        rows.append(row)
    return rows


def _lhs_values(
    rng: np.random.Generator,
    n_samples: int,
    bounds: dict[str, tuple[float, float]],
) -> list[dict[str, float]]:
    """Generate a small deterministic Latin-hypercube design for uncertainty sweeps."""
    unit: dict[str, np.ndarray] = {}
    for name in bounds:
        points = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(points)
        unit[name] = points

    rows: list[dict[str, float]] = []
    for idx in range(n_samples):
        row: dict[str, float] = {}
        for name, (low, high) in bounds.items():
            row[name] = float(low + unit[name][idx] * (high - low))
        rows.append(row)
    return rows


def _summarize_uncertainty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_skill = np.array([row["final_skill"] for row in rows], dtype=float)
    gap = np.array([row["quality_skill_gap"] for row in rows], dtype=float)
    crossings = [row["tipping_point_epoch"] for row in rows if row["tipping_point_epoch"] is not None]
    summary: dict[str, Any] = {
        "n_scenarios": len(rows),
        "crossing_share": round(len(crossings) / len(rows), 4),
        "no_crossing_count": len(rows) - len(crossings),
        "final_skill_p10_p50_p90": [round(float(v), 4) for v in np.quantile(final_skill, [0.1, 0.5, 0.9])],
        "gap_p10_p50_p90": [round(float(v), 4) for v in np.quantile(gap, [0.1, 0.5, 0.9])],
    }
    if crossings:
        crossing_arr = np.array(crossings, dtype=float)
        summary["crossing_epoch_p10_p50_p90"] = [
            round(float(v), 1) for v in np.quantile(crossing_arr, [0.1, 0.5, 0.9])
        ]
    else:
        summary["crossing_epoch_p10_p50_p90"] = None
    return summary


def run_lhs_uncertainty(seed: int = 42, n_samples: int = 64, n_monte_carlo: int = 60) -> dict[str, Any]:
    """Systematic multi-parameter uncertainty sweep for HOTL and HITL regimes."""
    rng = np.random.default_rng(seed)
    designs = {
        "loose": _lhs_values(
            rng,
            n_samples,
            {
                "alpha_atrophy": (0.005, 0.020),
                "red_queen_rate": (0.000, 0.010),
                "ai_interaction_floor": (0.25, 0.75),
                "base_engagement": (0.10, 0.50),
                "lazy_contagion_rate": (0.00, 0.45),
            },
        ),
        "strict": _lhs_values(
            rng,
            n_samples,
            {
                "alpha_atrophy": (0.005, 0.020),
                "red_queen_rate": (0.000, 0.010),
                "ai_interaction_floor": (0.25, 0.75),
                "base_engagement": (0.50, 0.90),
                "penalty": (0.05, 0.25),
            },
        ),
    }

    out: dict[str, Any] = {
        "n_samples": n_samples,
        "n_monte_carlo_per_scenario": n_monte_carlo,
        "ranges": {
            "loose": {
                "alpha_atrophy": [0.005, 0.020],
                "red_queen_rate": [0.000, 0.010],
                "ai_interaction_floor": [0.25, 0.75],
                "base_engagement": [0.10, 0.50],
                "lazy_contagion_rate": [0.00, 0.45],
                "penalty": "fixed at 0.00",
            },
            "strict": {
                "alpha_atrophy": [0.005, 0.020],
                "red_queen_rate": [0.000, 0.010],
                "ai_interaction_floor": [0.25, 0.75],
                "base_engagement": [0.50, 0.90],
                "penalty": [0.05, 0.25],
                "lazy_contagion_rate": "canonical 0.30",
            },
        },
        "samples": {},
        "summary": {},
    }

    for regime_key, rows in designs.items():
        sample_results: list[dict[str, Any]] = []
        for idx, design in enumerate(rows):
            params = PARAMS.copy()
            params["n_monte_carlo"] = n_monte_carlo
            params["alpha_atrophy"] = design["alpha_atrophy"]
            params["red_queen_rate"] = design["red_queen_rate"]
            params["ai_interaction_floor"] = design["ai_interaction_floor"]
            if regime_key == "loose":
                params["lazy_contagion_rate"] = design["lazy_contagion_rate"]
            regime = GOVERNANCE[regime_key].copy()
            regime["base_engagement"] = design["base_engagement"]
            if regime_key == "strict":
                regime["penalty"] = design["penalty"]

            row = summarize(aggregate(regime_key, regime, params, seed + idx + (0 if regime_key == "loose" else 1000)))
            row["scenario_id"] = idx
            row["parameters"] = {name: round(value, 5) for name, value in design.items()}
            sample_results.append(row)
        out["samples"][regime_key] = sample_results
        out["summary"][regime_key] = _summarize_uncertainty(sample_results)
    return out


def _summarize_paired_lhs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    loose_crossings = [row["loose"]["tipping_point_epoch"] for row in rows if row["loose"]["tipping_point_epoch"] is not None]
    strict_crossings = [row["strict"]["tipping_point_epoch"] for row in rows if row["strict"]["tipping_point_epoch"] is not None]
    skill_advantage = np.array(
        [row["strict"]["final_skill"] - row["loose"]["final_skill"] for row in rows],
        dtype=float,
    )
    gap_difference = np.array(
        [row["loose"]["quality_skill_gap"] - row["strict"]["quality_skill_gap"] for row in rows],
        dtype=float,
    )

    loose_earlier = 0
    for row in rows:
        loose_epoch = row["loose"]["tipping_point_epoch"]
        strict_epoch = row["strict"]["tipping_point_epoch"]
        loose_cmp = 10_000 if loose_epoch is None else int(loose_epoch)
        strict_cmp = 10_000 if strict_epoch is None else int(strict_epoch)
        if loose_cmp < strict_cmp:
            loose_earlier += 1

    return {
        "n_scenarios": len(rows),
        "loose_crossing_share": round(len(loose_crossings) / len(rows), 4),
        "strict_crossing_share": round(len(strict_crossings) / len(rows), 4),
        "loose_earlier_crossing_share": round(loose_earlier / len(rows), 4),
        "strict_final_skill_higher_share": round(float(np.mean(skill_advantage > 0)), 4),
        "loose_gap_larger_share": round(float(np.mean(gap_difference > 0)), 4),
        "strict_minus_loose_final_skill_p10_p50_p90": [
            round(float(v), 4) for v in np.quantile(skill_advantage, [0.1, 0.5, 0.9])
        ],
        "loose_minus_strict_gap_p10_p50_p90": [
            round(float(v), 4) for v in np.quantile(gap_difference, [0.1, 0.5, 0.9])
        ],
    }


def run_paired_lhs_common_conditions(
    seed: int = 42,
    n_samples: int = 64,
    n_monte_carlo: int = 30,
) -> dict[str, Any]:
    """Paired HOTL/HITL sweep over common structural assumptions.

    Unlike the regime-specific uncertainty sweep, this design uses the same
    environmental and technology draws for both regimes and keeps each regime's
    canonical governance definition fixed. It is a paired regime-ordering check,
    not a full calibration exercise.
    """
    rng = np.random.default_rng(seed + 20_000)
    designs = _lhs_values(
        rng,
        n_samples,
        {
            "alpha_atrophy": (0.005, 0.020),
            "red_queen_rate": (0.000, 0.010),
            "ai_interaction_floor": (0.25, 0.75),
            "lazy_contagion_rate": (0.00, 0.45),
        },
    )

    rows: list[dict[str, Any]] = []
    for idx, design in enumerate(designs):
        params = PARAMS.copy()
        params["n_monte_carlo"] = n_monte_carlo
        params.update(design)
        scenario_seed = seed + 30_000 + idx
        loose = summarize(aggregate("loose", GOVERNANCE["loose"].copy(), params, scenario_seed))
        strict = summarize(aggregate("strict", GOVERNANCE["strict"].copy(), params, scenario_seed))
        rows.append(
            {
                "scenario_id": idx,
                "parameters": {name: round(value, 5) for name, value in design.items()},
                "loose": loose,
                "strict": strict,
            }
        )

    return {
        "n_samples": n_samples,
        "n_monte_carlo_per_scenario": n_monte_carlo,
        "ranges": {
            "alpha_atrophy": [0.005, 0.020],
            "red_queen_rate": [0.000, 0.010],
            "ai_interaction_floor": [0.25, 0.75],
            "lazy_contagion_rate": [0.00, 0.45],
            "governance": "canonical HITL and HOTL settings held fixed",
        },
        "samples": rows,
        "summary": _summarize_paired_lhs(rows),
    }


def run_immediate_ai_removal_exposure(periods: tuple[int, ...] = (18, 46, 59)) -> dict[str, Any]:
    """Measure immediate output exposure if AI support is removed at diagnostic periods."""
    results_path = Path(__file__).parent / "simulation_results_canonical.json"
    canonical = json.loads(results_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for regime_key in ["loose", "strict"]:
        data = canonical[regime_key]
        for period in periods:
            skill = float(data["skills_mean"][period])
            quality = float(data["output_quality_mean"][period])
            rows.append(
                {
                    "regime": regime_key,
                    "period": period,
                    "augmented_quality": round(quality, 4),
                    "standalone_quality_if_ai_removed": round(skill, 4),
                    "immediate_quality_change": round(skill - quality, 4),
                    "quality_skill_gap": round(quality - skill, 4),
                }
            )
    return {
        "definition": "Immediate one-period AI-removal exposure sets output quality to contemporaneous standalone skill S_t.",
        "rows": rows,
    }


def main() -> None:
    output = {
        "seed": 42,
        "n_monte_carlo": 200,
        "mechanism_ablation": run_mechanism_ablation(seed=42),
        "red_queen_sweep": run_red_queen_sweep(seed=42),
        "structural_sensitivity": run_structural_sensitivity(seed=42),
        "joint_stress_tests": run_joint_stress_tests(seed=42),
        "lhs_uncertainty": run_lhs_uncertainty(seed=42),
        "paired_lhs_common_conditions": run_paired_lhs_common_conditions(seed=42),
        "immediate_ai_removal_exposure": run_immediate_ai_removal_exposure(),
    }
    out_path = Path(__file__).parent / "robustness_ablation_canonical.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
