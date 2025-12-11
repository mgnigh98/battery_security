import matplotlib.pyplot as plt
import numpy as np

font_size = 18
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (12, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1, 'lines.linewidth': 2.5, "axes.linewidth": 2.5,
          'axes.axisbelow': True}
# plt.subplots_adjust(left=0.153, right=0.98, top=0.695, bottom=0.126, hspace=0.2, wspace=0.2)
plt.rcParams.update(rc)

# Aspects
aspects = [
    "Weight constraints", "Power constraints", "Redundancy requirements",
    "Operational risk", "Thermal management", "Operation environment",
    "Battery swap risk", "Electromagnetic interference",
    "Cybersecurity risk", "Urban disruption impact"
]

# Values (1=Low, 2=Moderate, 3=High)
BESS = [1,1,2,1,1,1,1,1,1,2]
EVs  = [2,2,1,2,2,2,2,1,2,2]
eUAM = [3,3,3,3,3,3,3,3,3,3]

# Positioning
y = np.arange(len(aspects))
bar_height = 0.25

plt.barh(y - bar_height, BESS, height=bar_height, label="BESS")
plt.barh(y, EVs, height=bar_height, label="EVs")
plt.barh(y + bar_height, eUAM, height=bar_height, label="eUAM")

plt.yticks(y, aspects)
plt.xticks([1, 2, 3], ["Low", "Moderate", "High"])
plt.xlabel("Risk/Requirement Level", fontsize=22)
plt.title("Safety and Security Aspects: BESS vs EVs vs eUAM", fontsize=22)
# plt.legend()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
plt.tight_layout()
plt.savefig("./Figures/comp_gantt_chart.png")
plt.show()
