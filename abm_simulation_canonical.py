"""
Canonical ABM Simulation: The Atrophy-Dependency Cycle
======================================================

This script reconciles the two prior result streams:
- Phase 4: 1000-run team ABM with Red Queen threshold and team spillovers.
- Phase 9: engagement-weighted AI utility, adaptive engagement, dependency
  hysteresis, recovery scenario, and sensitivity analysis.

Outputs:
- simulation_results_canonical.json
- sensitivity_analysis_canonical.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore


PARAMS: dict[str, Any] = {
    "n_workers": 50,
    "n_epochs": 60,
    "n_monte_carlo": 1000,
    "alpha_atrophy": 0.01,
    "beta_dependency": 2.0,
    "ai_capability": 0.85,
    "learning_rate": 0.005,
    "task_min_complexity": 0.4,
    "red_queen_rate": 0.005,
    "skill_noise_std": 0.005,
    "quality_noise_std": 0.02,
    "initial_skill_mean": 0.6,
    "initial_skill_std": 0.1,
    "dep_memory": 0.3,
    "adaptive_engagement_rate": 0.05,
    "lazy_contagion_rate": 0.3,
    "knowledge_spillover_rate": 0.02,
    "ai_interaction_floor": 0.5,
    "ai_interaction_form": "linear",
    "adaptive_engagement_cap": 0.3,
}

GOVERNANCE: dict[str, dict[str, Any]] = {
    "baseline": {
        "base_engagement": 1.0,
        "ai_available": False,
        "penalty": 0.0,
        "label": "No AI (Baseline)",
    },
    "strict": {
        "base_engagement": 0.7,
        "ai_available": True,
        "penalty": 0.15,
        "label": "HITL (Strict)",
    },
    "loose": {
        "base_engagement": 0.1,
        "ai_available": True,
        "penalty": 0.0,
        "label": "HOTL (Loose)",
    },
    "recovery": {
        "base_engagement": 0.1,
        "ai_available": True,
        "penalty": 0.0,
        "label": "Recovery (HOTL to HITL at Epoch 30)",
        "switch_epoch": 30,
        "switch_to_engagement": 0.7,
        "switch_to_penalty": 0.15,
    },
}


def effective_regime(regime: dict[str, Any], epoch: int) -> tuple[float, float]:
    """Return base engagement and governance penalty for an epoch."""
    if "switch_epoch" in regime and epoch >= int(regime["switch_epoch"]):
        return float(regime["switch_to_engagement"]), float(regime["switch_to_penalty"])
    return float(regime["base_engagement"]), float(regime["penalty"])


def effective_ai_value(skill: float, params: dict[str, Any]) -> float:
    """Map human skill into extractable AI capability."""
    floor = float(params["ai_interaction_floor"])
    form = str(params.get("ai_interaction_form", "linear"))
    skill = float(np.clip(skill, 0.0, 1.0))
    if form == "concave":
        access = floor + (1.0 - floor) * np.sqrt(skill)
    elif form == "convex":
        access = floor + (1.0 - floor) * skill**2
    elif form == "threshold":
        access = floor if skill < 0.5 else 1.0
    else:
        access = floor + (1.0 - floor) * skill
    return float(params["ai_capability"]) * access


def simulate_team(
    regime: dict[str, Any],
    params: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate one interacting team under one governance regime."""
    n_workers = int(params["n_workers"])
    n_epochs = int(params["n_epochs"])

    skills = np.clip(
        rng.normal(params["initial_skill_mean"], params["initial_skill_std"], n_workers),
        0.0,
        1.0,
    )
    prev_skills = skills.copy()
    deps = np.zeros(n_workers)

    history_skills = np.zeros((n_epochs, n_workers))
    history_qualities = np.zeros((n_epochs, n_workers))
    history_deps = np.zeros((n_epochs, n_workers))
    history_engagements = np.zeros((n_epochs, n_workers))
    tau_history = np.zeros(n_epochs)

    avg_quality = float(np.mean(skills))

    for t in range(n_epochs):
        tau_history[t] = min(
            1.0,
            float(params["task_min_complexity"]) + float(params["red_queen_rate"]) * t,
        )
        base_eng, penalty = effective_regime(regime, t)
        avg_dep = float(np.mean(deps))

        qualities = np.zeros(n_workers)
        engagements = np.zeros(n_workers)

        for i in range(n_workers):
            if regime["ai_available"]:
                social_engagement = base_eng * (
                    1.0 - float(params["lazy_contagion_rate"]) * avg_dep
                )
                skill_decline = max(0.0, prev_skills[i] - skills[i])
                adaptive_boost = min(
                    float(params["adaptive_engagement_cap"]),
                    float(params["adaptive_engagement_rate"]) * skill_decline * 10.0,
                )
                engagement = np.clip(social_engagement + adaptive_boost, 0.01, 1.0)
            else:
                engagement = 1.0

            effective_ai = effective_ai_value(float(skills[i]), params)
            if regime["ai_available"]:
                raw_quality = engagement * skills[i] + (1.0 - engagement) * effective_ai - penalty
            else:
                raw_quality = skills[i]

            qualities[i] = np.clip(
                raw_quality + rng.normal(0, params["quality_noise_std"]),
                0.0,
                1.0,
            )
            engagements[i] = engagement

        history_skills[t] = skills
        history_qualities[t] = qualities
        history_deps[t] = deps
        history_engagements[t] = engagements

        new_skills = skills.copy()
        for i in range(n_workers):
            spillover = 0.0
            if avg_quality > skills[i]:
                spillover = float(params["knowledge_spillover_rate"]) * (avg_quality - skills[i])

            atrophy = float(params["alpha_atrophy"]) * (1.0 - engagements[i])
            growth = float(params["learning_rate"]) * engagements[i] + spillover
            new_skills[i] = np.clip(
                skills[i] - atrophy + growth + rng.normal(0, params["skill_noise_std"]),
                0.0,
                1.0,
            )

        if regime["ai_available"]:
            instant_deps = 1.0 - np.exp(
                -float(params["beta_dependency"]) * np.maximum(0.0, qualities - skills)
            )
            deps = float(params["dep_memory"]) * deps + (1.0 - float(params["dep_memory"])) * instant_deps
        else:
            deps = np.zeros(n_workers)

        prev_skills = skills
        skills = new_skills
        avg_quality = float(np.mean(qualities))

    return {
        "skills": history_skills,
        "qualities": history_qualities,
        "deps": history_deps,
        "engagements": history_engagements,
        "tau_history": tau_history,
    }


def aggregate(regime_key: str, regime: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Run Monte Carlo simulations and aggregate team-level trajectories."""
    rng = np.random.default_rng(seed)
    n_mc = int(params["n_monte_carlo"])
    n_epochs = int(params["n_epochs"])

    all_skills = np.zeros((n_mc, n_epochs))
    all_quality = np.zeros((n_mc, n_epochs))
    all_deps = np.zeros((n_mc, n_epochs))
    all_engagements = np.zeros((n_mc, n_epochs))
    tau_history: np.ndarray | None = None

    for run in range(n_mc):
        result = simulate_team(regime, params, rng)
        all_skills[run] = result["skills"].mean(axis=1)
        all_quality[run] = result["qualities"].mean(axis=1)
        all_deps[run] = result["deps"].mean(axis=1)
        all_engagements[run] = result["engagements"].mean(axis=1)
        if tau_history is None:
            tau_history = result["tau_history"]

    assert tau_history is not None
    summary = {
        "governance": regime["label"],
        "skills_mean": all_skills.mean(axis=0).tolist(),
        "skills_std": all_skills.std(axis=0).tolist(),
        "output_quality_mean": all_quality.mean(axis=0).tolist(),
        "output_quality_std": all_quality.std(axis=0).tolist(),
        "ai_dependency_mean": all_deps.mean(axis=0).tolist(),
        "engagement_mean": all_engagements.mean(axis=0).tolist(),
        "tau_history": tau_history.tolist(),
        "n_runs": params["n_monte_carlo"],
        "n_workers": params["n_workers"],
        "t_months": params["n_epochs"],
        "regime_key": regime_key,
    }
    summary["tipping_point"] = find_tipping_point(summary)
    return summary


def find_tipping_point(result: dict[str, Any]) -> dict[str, Any]:
    """Find first epoch where team mean skill is below the contemporaneous threshold."""
    for epoch, (skill, tau) in enumerate(zip(result["skills_mean"], result["tau_history"])):
        if skill < tau:
            return {"first_crossing_epoch": epoch, "threshold_at_crossing": tau}
    return {"first_crossing_epoch": None, "threshold_at_crossing": None}


def run_main_results(seed: int = 42) -> dict[str, Any]:
    return {
        key: aggregate(key, regime, PARAMS.copy(), seed)
        for key, regime in GOVERNANCE.items()
    }


def run_sensitivity(seed: int = 42) -> dict[str, Any]:
    """Sensitivity sweep on HOTL; fewer runs are used for tractability."""
    sweep_values = {
        "alpha_atrophy": np.linspace(0.005, 0.020, 7),
        "beta_dependency": np.linspace(1.0, 3.0, 7),
        "learning_rate": np.linspace(0.0025, 0.010, 7),
    }
    out: dict[str, Any] = {}
    for param_name, values in sweep_values.items():
        rows: list[dict[str, Any]] = []
        for value in values:
            params = PARAMS.copy()
            params["n_monte_carlo"] = 200
            params[param_name] = float(value)
            result = aggregate("loose", GOVERNANCE["loose"], params, seed)
            final_skill = result["skills_mean"][-1]
            final_quality = result["output_quality_mean"][-1]
            rows.append(
                {
                    "param_value": round(float(value), 4),
                    "tipping_point_epoch": result["tipping_point"]["first_crossing_epoch"],
                    "final_skill": round(final_skill, 4),
                    "final_quality": round(final_quality, 4),
                    "quality_skill_gap": round(final_quality - final_skill, 4),
                }
            )
        out[param_name] = rows
    return out


def write_outputs() -> None:
    project_root = Path(__file__).parent.parent
    out_dir = project_root
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_main_results(seed=42)
    sensitivity = run_sensitivity(seed=42)

    (out_dir / "simulation_results_canonical.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (out_dir / "sensitivity_analysis_canonical.json").write_text(
        json.dumps(sensitivity, indent=2),
        encoding="utf-8",
    )

    print("Canonical results written:")
    print(f"  {out_dir / 'simulation_results_canonical.json'}")
    print(f"  {out_dir / 'sensitivity_analysis_canonical.json'}")
    for key, result in results.items():
        s0 = result["skills_mean"][0]
        s1 = result["skills_mean"][-1]
        q1 = result["output_quality_mean"][-1]
        gap = q1 - s1
        tp = result["tipping_point"]["first_crossing_epoch"]
        print(
            f"{key:8s} S {s0:.3f}->{s1:.3f}, Q {q1:.3f}, gap {gap:+.3f}, "
            f"TP {tp}"
        )


if __name__ == "__main__":
    write_outputs()
