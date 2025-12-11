#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eUAM mission-phase severity bars (Safety vs Security)
- Bars instead of lines
- Two separate panels (Safety on top, Security below)
- Legend outside plot area
- Saves PNG and PDF next to the script
"""

import matplotlib.pyplot as plt
import numpy as np


font_size = 18
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (14, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1, 'lines.linewidth': 2.5, "axes.linewidth": 2.5,
          'axes.axisbelow': True}
# plt.subplots_adjust(left=0.153, right=0.98, top=0.695, bottom=0.126, hspace=0.2, wspace=0.2)
plt.rcParams.update(rc)
# -----------------------------
# 1) Define mission phases
# -----------------------------
phases = ["Take-off / Climb", "Cruise", "Landing / Descent"]
x = np.arange(len(phases))  # 0,1,2

# Severity scale: 1=Low, 2=Moderate, 3=High (you can refine to 1..5 if needed)
# SAFETY
power       = np.array([3, 2, 2])  # burst at take-off, sustained in cruise, medium at landing
thermal     = np.array([3, 2, 1])  # rapid heating at take-off, moderate in cruise, lower at landing
redundancy  = np.array([3, 2, 3])  # critical at take-off, moderate cruise, high for safe landing
# SECURITY
security    = np.array([2, 2, 3])  # comm/control at take-off, steady in cruise, highest at landing (GPS/nav)

# -----------------------------
# 2) Plot settings
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (14, 10),
    "font.size": 14
})

fig, (ax_safety, ax_sec) = plt.subplots(
    nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

# ---- Helper to draw grouped bars along x (phases) ----
def grouped_bars(ax, series_list, labels, width=0.22):
    """
    series_list: list of 1D arrays (same length as phases)
    labels: legend labels for series_list
    width: bar width
    """
    n_series = len(series_list)
    # offsets centered around x
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width
    bar_containers = []
    for s, lab, dx in zip(series_list, labels, offsets):
        bc = ax.bar(x + dx, s, width=width, label=lab)
        bar_containers.append(bc)
    return bar_containers

# -----------------------------
# 3) SAFETY panel (top)
# -----------------------------
_ = grouped_bars(
    ax_safety,
    series_list=[power, thermal, redundancy],
    labels=["Power demand", "Thermal challenge", "Redundancy requirement"],
    width=0.22
)

ax_safety.set_ylim(0.6, 3.4)
ax_safety.set_yticks([1, 2, 3], ["Low", "Moderate", "High"])
ax_safety.set_ylabel("Relative severity", fontsize=18)
ax_safety.set_title("Safety across mission phases", fontsize=20)

# Grid + legend outside
ax_safety.grid(True, axis="y", linestyle="--", alpha=0.4)
ax_safety.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12, frameon=False)

# Optional annotations (edit/remove as you like)
ax_safety.annotate("Burst power\n$>\sim 200$ W/kg (indic.)",
                   xy=(x[0], power[0]), xycoords="data",
                   xytext=(90, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", lw=1),
                   ha="right", va="bottom")

ax_safety.annotate("Sustained load\n$\sim 60–70\%$ peak (indic.)",
                   xy=(x[1], power[1]), xycoords="data",
                   xytext=(5, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", lw=1),
                   ha="left", va="bottom")

# -----------------------------
# 4) SECURITY panel (bottom)
# -----------------------------
_ = grouped_bars(
    ax_sec,
    series_list=[security],
    labels=["Security risk"],
    width=0.22
)

ax_sec.set_ylim(0.6, 3.4)
ax_sec.set_yticks([1, 2, 3], ["Low", "Moderate", "High"])
ax_sec.set_xlabel("Mission phase", fontsize=18)
ax_sec.set_ylabel("Relative severity", fontsize=18)
ax_sec.set_title("Security across mission phases", fontsize=20)

# x-ticks (shared across panels)
ax_sec.set_xticks(x, phases, rotation=0)

# Grid + legend outside
ax_sec.grid(True, axis="y", linestyle="--", alpha=0.4)
ax_sec.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=14, frameon=False)

# Optional annotation
ax_sec.annotate("Navigation/comm spoofing risk ↑",
                xy=(x[2], security[2]), xycoords="data",
                xytext=(-180, 2), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1),
                ha="left", va="bottom")

# -----------------------------
# 5) Layout & save
# -----------------------------
fig.suptitle("eUAM Battery Safety and Security - Mission-Phase Severity", y=0.98, fontsize=20)
fig.tight_layout()
# fig.tight_layout(rect=[0, 0, 0.84, 0.96])  # leave room on right for legends
plt.show()
fig.savefig("./Figures/mission_phase_bars.png", dpi=300)
fig.savefig("./Figures/mission_phase_bars.pdf")  # vector for papers
# print("Saved: mission_phase_bars.png and mission_phase_bars.pdf")
