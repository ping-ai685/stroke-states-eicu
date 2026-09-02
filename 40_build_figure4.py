"""
Paper 2: Figure 4 — one-step-ahead prediction (Aim 4).

Three panels, each carrying one claim the text makes:

  A  the ladder at six hours, in both databases. Shows that every addition of
     information helps, that the null is at 0.5 by construction, and that the
     internal and external values sit on top of each other.
  B  the horizon curves in eICU. Shows the finding that matters most: the frozen
     representation and the sequence model converge, because the first improves
     steeply with the horizon and the second barely does. The shaded band is what
     the four states discard, and it closes.
  C  average precision against the base rate. The honesty panel: enrichment over
     chance is roughly twofold, so none of this is a usable alarm.

Style follows step 26 so the four figures read as one set.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from figure_labels import labels, FONT                    # noqa: E402

LANG = sys.argv[1].upper() if len(sys.argv) > 1 else "EN"
T = labels(LANG)
PRED = HERE / "prediction"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DPI = 300
plt.rcParams.update({"font.family": FONT[LANG], "font.size": 8.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "axes.unicode_minus": False})
INK, MIM, EIC, MUT = "#1a1a1a", "#8a8a8a", "#0F5C63", "#c0c0c0"
SEQ_C, WIN_C = "#B45309", "#7C7C7C"

M = pd.read_csv(PRED / "metrics_by_model.csv")
L6 = pd.read_csv(PRED / "metrics_L6.csv")
MH = pd.read_csv(PRED / "multihorizon_metrics.csv")

LADDER = [(k, T[k]) for k in ("L0", "L1", "L2", "L3", "L4", "L5", "L6")]

fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1, 1],
              wspace=0.42, hspace=0.55, figure=fig)

# ---------------------------------------------------------------- A: the ladder
axA = fig.add_subplot(gs[:, 0])
y = np.arange(len(LADDER))[::-1]
for off, scope, col, lab in [(+0.17, "mimic-test", MIM, T["f4_mimic"]),
                             (-0.17, "eicu-full", EIC, T["f4_eicu"])]:
    v, lo, hi = [], [], []
    for k, _ in LADDER:
        if k == "L6":
            r = L6[(L6.scope == scope) & (L6.horizon_h == 6)].iloc[0]
            v.append(r.change_auroc); lo.append(r.change_auroc_lo); hi.append(r.change_auroc_hi)
        else:
            r = M[(M.scope == scope) & (M.model == k)].iloc[0]
            v.append(r.change_auroc); lo.append(r.change_auroc_lo); hi.append(r.change_auroc_hi)
    v, lo, hi = np.array(v), np.array(lo), np.array(hi)
    axA.hlines(y + off, lo, hi, color=col, linewidth=1.5, alpha=.85, zorder=2)
    axA.scatter(v, y + off, s=17, color=col, zorder=3, linewidth=0, label=lab)

axA.axvline(0.5, color=MUT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
axA.text(0.503, y[0] + 0.52, T["f4_nodisc"], fontsize=6.4, color="#777777", va="bottom")
axA.set_yticks(y); axA.set_yticklabels([n for _, n in LADDER], fontsize=7.2)
axA.set_xlim(0.455, 0.84); axA.set_ylim(-0.85, len(LADDER) - 0.25)
axA.set_xlabel(T["f4_xlabel"], fontsize=7.4)
axA.legend(fontsize=6.6, frameon=False, loc="lower left", handletextpad=.4,
           bbox_to_anchor=(0.005, 0.0))
axA.set_title(T["f4_A"], fontsize=8.2, loc="left",
              fontweight="bold", pad=6)
for s in ("top", "right"): axA.spines[s].set_visible(False)
axA.axhspan(y[3] + 0.5, y[0] + 0.7, color="#f2f2f2", zorder=0)
axA.text(0.475, y[1], T["f4_nofit"], fontsize=6.4, color="#999999", rotation=90,
         va="center", ha="center")

# ------------------------------------------------------- B: convergence by horizon
axB = fig.add_subplot(gs[0, 1])
H = [6, 12, 24]
l2 = [MH[(MH.scope == "eicu-full") & (MH.horizon_h == h) & (MH.model == "L2")].change_auroc.iloc[0] for h in H]
win = [MH[(MH.scope == "eicu-full") & (MH.horizon_h == h) & (MH.model.isin(["L3", "L4", "L5"]))].change_auroc.max() for h in H]
l6 = [L6[(L6.scope == "eicu-full") & (L6.horizon_h == h)].change_auroc.iloc[0] for h in H]
x = np.arange(len(H))
axB.fill_between(x, l2, l6, color=SEQ_C, alpha=.11, zorder=1)
axB.set_xlim(-0.32, len(H) - 0.68)
axB.plot(x, l6, "-o", color=SEQ_C, markersize=3.6, linewidth=1.3, label=T["f4_seq"], zorder=3)
axB.plot(x, win, "-o", color=WIN_C, markersize=3.0, linewidth=1.0, label=T["f4_win"], zorder=3)
axB.plot(x, l2, "-o", color=EIC, markersize=3.6, linewidth=1.3, label=T["f4_frozen"], zorder=3)
axB.axhline(0.5, color=MUT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
for i, h in enumerate(H):
    share = (l2[i] - .5) / (l6[i] - .5) * 100
    axB.annotate(f"{share:.0f}%", (x[i], l2[i] - 0.022), fontsize=6.5, color=EIC,
                 ha="center", va="top", fontweight="bold")
axB.text(0.5, 1.0, T["f4_share"],
         transform=axB.transAxes, fontsize=6.0, color="#999999", ha="center", va="top")
axB.set_xticks(x); axB.set_xticklabels([f"{h}{T[chr(34)+chr(34)] if False else T['hr']}" for h in H], fontsize=7.2)
axB.set_ylim(0.48, 0.87); axB.set_ylabel(T["f4_auroc"], fontsize=7.4)
axB.legend(fontsize=6.2, frameon=False, loc="lower right", handletextpad=.4, borderpad=.2)
axB.set_title(T["f4_B"], fontsize=8.2, loc="left",
              fontweight="bold", pad=6)
for s in ("top", "right"): axB.spines[s].set_visible(False)

# --------------------------------------------------- C: precision against base rate
axC = fig.add_subplot(gs[1, 1])
base = [MH[(MH.scope == "eicu-full") & (MH.horizon_h == h) & (MH.model == "L2")].changed_pct.iloc[0] / 100 for h in H]
ap2 = [MH[(MH.scope == "eicu-full") & (MH.horizon_h == h) & (MH.model == "L2")].change_auprc.iloc[0] for h in H]
ap6 = [L6[(L6.scope == "eicu-full") & (L6.horizon_h == h)].change_auprc.iloc[0] for h in H]
w = 0.27
axC.bar(x - w, base, w, color=MUT, label=T["f4_base"])
axC.bar(x, ap2, w, color=EIC, label=T["f4_frozen"])
axC.bar(x + w, ap6, w, color=SEQ_C, label=T["f4_seq"])
for i in range(len(H)):
    axC.text(x[i], ap2[i] - .022, f"{ap2[i]/base[i]:.1f}×", ha="center", va="top",
             fontsize=6.2, color="white", fontweight="bold")
axC.set_xticks(x); axC.set_xticklabels([f"{h}{T['hr']}" for h in H], fontsize=7.2)
axC.set_ylim(0, 0.60); axC.set_ylabel(T["f4_ap"], fontsize=7.4)
axC.legend(fontsize=6.2, frameon=False, loc="upper left", handletextpad=.4, borderpad=.2)
axC.set_title(T["f4_C"], fontsize=8.2, loc="left",
              fontweight="bold", pad=6)
for s in ("top", "right"): axC.spines[s].set_visible(False)

for ext in ("png", "tif"):
    fig.savefig(FIG / f"figure4_prediction_{LANG}.{ext}", dpi=DPI, bbox_inches="tight",
                facecolor="white")
print(f"wrote {FIG/('figure4_prediction_' + LANG + '.png')} and .tif")
print(f"  panel A: {len(LADDER)} predictors x 2 databases")
print(f"  panel B: frozen {l2[0]:.3f}->{l2[-1]:.3f}, sequence {l6[0]:.3f}->{l6[-1]:.3f}, "
      f"share {(l2[0]-.5)/(l6[0]-.5)*100:.1f}% -> {(l2[-1]-.5)/(l6[-1]-.5)*100:.1f}%")
print(f"  panel C: base rate {base[0]:.3f}->{base[-1]:.3f}, frozen AP {ap2[0]:.3f}->{ap2[-1]:.3f}")
