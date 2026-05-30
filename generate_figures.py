#!/usr/bin/env python3
"""
Figure generation for "The Atrophy-Dependency Cycle" paper.
Input: simulation_results_canonical.json
Output: fig1_skill_trajectory.pdf ... fig4_dependency_growth.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "simulation_results_canonical.json"
OUT_DIR = Path(__file__).parent
os.makedirs(OUT_DIR, exist_ok=True)

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette: accessible and publication-quality
C_BASELINE = '#2ca02c'    # Green
C_HITL     = '#1f77b4'    # Blue
C_HOTL     = '#d62728'    # Red
C_TIPPING  = '#ff7f0e'    # Orange

# Load data

with open(DATA_FILE) as f:
    data = json.load(f)

epochs = np.arange(len(data['baseline']['skills_mean']))

baseline_skill = np.array(data['baseline']['skills_mean'])
baseline_skill_std = np.array(data['baseline']['skills_std'])
baseline_quality = np.array(data['baseline']['output_quality_mean'])
baseline_quality_std = np.array(data['baseline']['output_quality_std'])

hitl_skill = np.array(data['strict']['skills_mean'])
hitl_skill_std = np.array(data['strict']['skills_std'])
hitl_quality = np.array(data['strict']['output_quality_mean'])
hitl_quality_std = np.array(data['strict']['output_quality_std'])
hitl_dep = np.array(data['strict']['ai_dependency_mean'])

hotl_skill = np.array(data['loose']['skills_mean'])
hotl_skill_std = np.array(data['loose']['skills_std'])
hotl_quality = np.array(data['loose']['output_quality_mean'])
hotl_quality_std = np.array(data['loose']['output_quality_std'])
hotl_dep = np.array(data['loose']['ai_dependency_mean'])

tau_history = np.array(data['baseline']['tau_history'])

# Find tipping points (where skill < tau_history)
hitl_crosses = hitl_skill < tau_history
hotl_crosses = hotl_skill < tau_history

hitl_tipping_idx = np.argmax(hitl_crosses) if np.any(hitl_crosses) else None
hotl_tipping_idx = np.argmax(hotl_crosses) if np.any(hotl_crosses) else None

print(f"Data loaded: {len(epochs)} epochs")
print(f"HITL Tipping point: Epoch {hitl_tipping_idx}" if hitl_tipping_idx else "No HITL tipping point found")
print(f"HOTL Tipping point: Epoch {hotl_tipping_idx}" if hotl_tipping_idx else "No HOTL tipping point found")

# FIGURE 1: Skill Trajectory Across Three Governance Regimes

fig1, ax1 = plt.subplots(figsize=(8, 5))

# Confidence bands (±1 SD)
ax1.fill_between(epochs, baseline_skill - baseline_skill_std, baseline_skill + baseline_skill_std,
                 color=C_BASELINE, alpha=0.15)
ax1.fill_between(epochs, hitl_skill - hitl_skill_std, hitl_skill + hitl_skill_std,
                 color=C_HITL, alpha=0.15)
ax1.fill_between(epochs, hotl_skill - hotl_skill_std, hotl_skill + hotl_skill_std,
                 color=C_HOTL, alpha=0.15)

# Mean lines
ax1.plot(epochs, baseline_skill, color=C_BASELINE, linewidth=2.2, label='Baseline (No AI)', zorder=5)
ax1.plot(epochs, hitl_skill, color=C_HITL, linewidth=2.2, label='HITL (Strict)', linestyle='--', zorder=5)
ax1.plot(epochs, hotl_skill, color=C_HOTL, linewidth=2.2, label='HOTL (Loose)', linestyle='-', zorder=5)

# Dynamic Red Queen threshold
ax1.plot(epochs, tau_history, color='gray', linestyle=':', alpha=0.8, linewidth=1.5, label='Min. Task Complexity ($\\tau_t$)')

# HOTL Tipping point annotation
if hotl_tipping_idx:
    ax1.axvline(x=hotl_tipping_idx, color=C_HOTL, linestyle='--', alpha=0.5, linewidth=1.5)

# HITL Tipping point annotation
if hitl_tipping_idx:
    ax1.axvline(x=hitl_tipping_idx, color=C_HITL, linestyle='--', alpha=0.5, linewidth=1.5)

# Final values annotation
for label, val, color, y_off in [
    ('Baseline', baseline_skill[-1], C_BASELINE, 0.01),
    ('HITL', hitl_skill[-1], C_HITL, -0.02),
    ('HOTL', hotl_skill[-1], C_HOTL, 0.02),
]:
    ax1.annotate(f'{val:.3f}', xy=(epochs[-1], val), xytext=(epochs[-1] + 1.5, val + y_off),
                 fontsize=9, color=color, fontweight='bold')

ax1.set_xlabel('Simulation period')
ax1.set_ylabel('Baseline Skill ($S_t$)')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_xlim(-1, epochs[-1] + 8)
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3, linestyle='-')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fig1.tight_layout()
fig1.savefig(OUT_DIR / 'fig1_skill_trajectory.pdf')
fig1.savefig(OUT_DIR / 'fig1_skill_trajectory.png')
print("Figure 1 saved: fig1_skill_trajectory.pdf/png")
plt.close(fig1)

# FIGURE 2: Output Quality Trajectory

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.fill_between(epochs, baseline_quality - baseline_quality_std, baseline_quality + baseline_quality_std,
                 color=C_BASELINE, alpha=0.15)
ax2.fill_between(epochs, hitl_quality - hitl_quality_std, hitl_quality + hitl_quality_std,
                 color=C_HITL, alpha=0.15)
ax2.fill_between(epochs, hotl_quality - hotl_quality_std, hotl_quality + hotl_quality_std,
                 color=C_HOTL, alpha=0.15)

ax2.plot(epochs, baseline_quality, color=C_BASELINE, linewidth=2.2, label='Baseline (No AI)', zorder=5)
ax2.plot(epochs, hitl_quality, color=C_HITL, linewidth=2.2, label='HITL (Strict)', linestyle='--', zorder=5)
ax2.plot(epochs, hotl_quality, color=C_HOTL, linewidth=2.2, label='HOTL (Loose)', linestyle='-', zorder=5)
ax2.plot(epochs, hotl_skill, color=C_HOTL, linewidth=1.8, label='HOTL standalone skill', linestyle=':', alpha=0.85, zorder=4)
ax2.fill_between(epochs, hotl_skill, hotl_quality, where=hotl_quality >= hotl_skill,
                 color=C_HOTL, alpha=0.08, label='_nolegend_')

for label, val, color in [
    ('Baseline', baseline_quality[-1], C_BASELINE),
    ('HITL', hitl_quality[-1], C_HITL),
    ('HOTL', hotl_quality[-1], C_HOTL),
]:
    ax2.annotate(f'{val:.3f}', xy=(epochs[-1], val), xytext=(epochs[-1] + 1.5, val),
                 fontsize=9, color=color, fontweight='bold')
ax2.annotate(f'Skill {hotl_skill[-1]:.3f}', xy=(epochs[-1], hotl_skill[-1]), xytext=(epochs[-1] + 1.5, hotl_skill[-1]),
             fontsize=9, color=C_HOTL, fontweight='bold')

ax2.set_xlabel('Simulation period')
ax2.set_ylabel('Output Quality / Standalone Skill')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.set_xlim(-1, epochs[-1] + 8)
ax2.set_ylim(0.25, 1.0)
ax2.grid(True, alpha=0.3, linestyle='-')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig(OUT_DIR / 'fig2_quality_trajectory.pdf')
fig2.savefig(OUT_DIR / 'fig2_quality_trajectory.png')
print("Figure 2 saved: fig2_quality_trajectory.pdf/png")
plt.close(fig2)

# FIGURE 3: Quality-Skill Gap

fig3, ax3 = plt.subplots(figsize=(8, 5))

gap_baseline = baseline_quality - baseline_skill
gap_hitl = hitl_quality - hitl_skill
gap_hotl = hotl_quality - hotl_skill

ax3.plot(epochs, gap_baseline, color=C_BASELINE, linewidth=2.2, label='Baseline (No AI)', zorder=5)
ax3.plot(epochs, gap_hitl, color=C_HITL, linewidth=2.2, label='HITL (Strict)', linestyle='--', zorder=5)
ax3.plot(epochs, gap_hotl, color=C_HOTL, linewidth=2.2, label='HOTL (Loose)', linestyle='-', zorder=5)

# Fill the positive HOTL output-skill gap
ax3.fill_between(epochs, 0, gap_hotl, color=C_HOTL, alpha=0.12, label='_nolegend_')

# Annotate the massive gap
if hotl_tipping_idx:
    ax3.axvline(x=hotl_tipping_idx, color=C_HOTL, linestyle='--', alpha=0.5, linewidth=1.5)
                 
if hitl_tipping_idx:
    ax3.axvline(x=hitl_tipping_idx, color=C_HITL, linestyle='--', alpha=0.5, linewidth=1.5)

# Final gap values
ax3.annotate(f'{gap_hotl[-1]:.3f}',
             xy=(epochs[-1], gap_hotl[-1]), xytext=(epochs[-1] + 0.8, gap_hotl[-1]),
             fontsize=9, color=C_HOTL, fontweight='bold')

ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.4)

ax3.set_xlabel('Simulation period')
ax3.set_ylabel('Quality-Skill Gap ($Q_t - S_t$)')
ax3.legend(loc='upper left', framealpha=0.9)
ax3.set_xlim(-1, epochs[-1] + 3)
ax3.grid(True, alpha=0.3, linestyle='-')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

fig3.tight_layout()
fig3.savefig(OUT_DIR / 'fig3_information_asymmetry.pdf')
fig3.savefig(OUT_DIR / 'fig3_information_asymmetry.png')
print("Figure 3 saved: fig3_information_asymmetry.pdf/png")
plt.close(fig3)

# FIGURE 4: Dependency Growth

fig4, ax4 = plt.subplots(figsize=(8, 5))

baseline_dep = np.zeros(len(epochs))  # No AI = no dependency
ax4.plot(epochs, baseline_dep, color=C_BASELINE, linewidth=2.2, label='Baseline (No AI)', zorder=5)
ax4.plot(epochs, hitl_dep, color=C_HITL, linewidth=2.2, label='HITL (Strict)', linestyle='--', zorder=5)
ax4.plot(epochs, hotl_dep, color=C_HOTL, linewidth=2.2, label='HOTL (Loose)', linestyle='-', zorder=5)

# HOTL threshold-crossing annotation
if hotl_tipping_idx is not None:
    ax4.axvline(x=hotl_tipping_idx, color=C_HOTL, linestyle='--', alpha=0.5, linewidth=1.5)

# HITL can cross the rising skill threshold late in the simulation, but its
# dependency remains near zero; avoid labeling that crossing as lock-in here.

# Final values
for label, arr, color in [
    ('HITL', hitl_dep, C_HITL),
    ('HOTL', hotl_dep, C_HOTL),
]:
    ax4.annotate(f'{arr[-1]:.3f}', xy=(epochs[-1], arr[-1]), xytext=(epochs[-1] + 1.5, arr[-1]),
                 fontsize=9, color=color, fontweight='bold')

ax4.set_xlabel('Simulation period')
ax4.set_ylabel('AI Dependency Rate ($D_t$)')
ax4.legend(loc='center left', framealpha=0.9)
ax4.set_xlim(-1, epochs[-1] + 8)
ax4.set_ylim(-0.02, 0.55)
ax4.grid(True, alpha=0.3, linestyle='-')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig4.tight_layout()
fig4.savefig(OUT_DIR / 'fig4_dependency_growth.pdf')
fig4.savefig(OUT_DIR / 'fig4_dependency_growth.png')
print("Figure 4 saved: fig4_dependency_growth.pdf/png")
plt.close(fig4)

print("\nAll 4 figures generated successfully")
print("Output directory: 6_Figures")
for f in sorted(OUT_DIR.glob("fig*")):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
