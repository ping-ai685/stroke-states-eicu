"""
Paper 2: build the three manuscript figures.

Figure 1  eICU cohort flow, including the two hospital eligibility rules.
Figure 2  Frozen-model transport: per-state profile agreement and prevalence.
Figure 3  De novo eICU structure: correspondence of the three- and four-state
          solutions to the discovery states.

English only for now; the Chinese set is generated after the English text is
final, so the two cannot drift the way the v1.0 protocol documents did.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import sys, torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from figure_labels import labels, FONT                    # noqa: E402

LANG = sys.argv[1].upper() if len(sys.argv) > 1 else "EN"
T = labels(LANG)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DPI = 300
plt.rcParams.update({"font.family": FONT[LANG], "font.size": 8.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "axes.unicode_minus": False})
INK, MIM, EIC, MUT = "#1a1a1a", "#8a8a8a", "#0F5C63", "#c0c0c0"
P = "neurologically preserved-low support"
R = "neurological impairment-respiratory support"
N = "neurological impairment-renal dysfunction"
L = "neurological impairment-low support"
ORDER = [P, L, N, R]
LAB = {P: T["st_P"], L: T["st_L"], N: T["st_N"], R: T["st_R"]}
FLAT = {k: v.replace("–\n", "–") for k, v in LAB.items()}

# =====================================================================
# Figure 1 — cohort flow
# =====================================================================
flow = pd.read_csv(HERE / "cohort/cohort_flow_counts.csv")
fig, ax = plt.subplots(figsize=(7.2, 9.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
MAIN_X, MAIN_W, EX_X, EX_W = 8, 52, 64, 33

def box(x, y, w, h, text, bold=False, fill="#ffffff", edge=INK, fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                                linewidth=0.9, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.45)

def arrow(y0, y1, x=MAIN_X + MAIN_W / 2):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.9, shrinkA=0, shrinkB=0))

STEPS = [
    (T["f1_source"], None),
    (T["f1_pheno"].format(n=int(flow.n_stays[0]), h=int(flow.n_hospitals[0])), None),
    (T["f1_stays"].format(n=int(flow.n_stays[1])),
     T["f1_x_trauma"].format(n=int(flow.n_stays[0] - flow.n_stays[1]))),
    (T["f1_stays"].format(n=int(flow.n_stays[2])),
     T["f1_x_age"].format(n=int(flow.n_stays[1] - flow.n_stays[2]))),
    (T["f1_stays"].format(n=int(flow.n_stays[3])),
     T["f1_x_first"].format(n=int(flow.n_stays[2] - flow.n_stays[3]))),
    (T["f1_stays"].format(n=int(flow.n_stays[4])),
     T["f1_x_12h"].format(n=int(flow.n_stays[3] - flow.n_stays[4]))),
    (T["f1_stays"].format(n=int(flow.n_stays[5])),
     T["f1_x_second"].format(n=int(flow.n_stays[4] - flow.n_stays[5]))),
]
y, H = 97, 7.8
for i, (main, excl) in enumerate(STEPS):
    box(MAIN_X, y, MAIN_W, H, main, bold=(i == 0))
    if excl:
        box(EX_X, y - 0.6, EX_W, H - 1.4, excl, fill="#f4f4f4", edge="#8a8a8a", fs=7.6)
        ax.annotate("", xy=(EX_X, y - H / 2), xytext=(MAIN_X + MAIN_W, y - H / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", linewidth=0.8))
    if i < len(STEPS) - 1:
        arrow(y - H, y - H - 2.6)
    y -= H + 2.6

box(MAIN_X, y, MAIN_W, H + 1.5,
    T["f1_cohort"],
    bold=True, fill="#e8f1f2", edge=EIC)
y_bot = y - H - 1.5
y2 = y_bot - 4.0
CX = [17, 50, 83]
ax.plot([MAIN_X + MAIN_W / 2, MAIN_X + MAIN_W / 2], [y_bot, y2 + 1.6], color=INK, linewidth=0.9)
ax.plot([CX[0], CX[2]], [y2 + 1.6, y2 + 1.6], color=INK, linewidth=0.9)
for cx in CX:
    ax.annotate("", xy=(cx, y2), xytext=(cx, y2 + 1.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.9, shrinkA=0, shrinkB=0))
box(CX[0] - 15, y2, 30, 10, T["f1_gcs"], fs=7.4)
box(CX[1] - 15, y2, 30, 10, T["f1_vent"], fs=7.4)
box(CX[2] - 15, y2, 30, 10, T["f1_both"],
    fs=7.4, fill="#e8f1f2", edge=EIC)
ax.text(50, y2 - 12.4, T["f1_rules"],
        ha="center", va="center", fontsize=7.4, style="italic", color="#555555")
fig.savefig(FIG / f"figure1_cohort_flow_{LANG}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / f"figure1_cohort_flow_{LANG}.tif", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig); print(f"Figure 1 written ({LANG})")

# =====================================================================
# Figure 2 — transport
# =====================================================================
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
CONT = [f for f in FEATS if f.endswith("_z")]
lblmap = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")["name"].to_dict()
mdl = torch.load(ROOT / ".model_k4.pt", weights_only=False)
for d in mdl.distributions:
    for s in d.distributions[-4:]:
        p = torch.clamp(s.probs.detach().clone(), 1e-4, 1 - 1e-4); p = p / p.sum(-1, keepdim=True)
        s.probs = torch.nn.Parameter(p, requires_grad=False); s._log_probs = torch.log(p)
mw = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(["stay_id", "window_idx"])
mw["state"] = pd.Series(np.concatenate(
    [mdl.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
     for _, g in mw.groupby("stay_id", sort=False)]), index=mw.index).map(lblmap)
MZ = mw.groupby("state")[FEATS].mean()
a = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
mo = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv")
el = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
bi = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
ew = mo.merge(bi, on=["patientunitstayid", "window_idx"]).merge(
    a[["patientunitstayid", "window_idx", "state", "vent_ascertainable"]],
    on=["patientunitstayid", "window_idx"])
sub = ew[ew.patientunitstayid.isin(el) & (ew.vent_ascertainable == 1)]
EZ = sub.groupby("state")[FEATS].mean()

fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.68, wspace=0.40)
for i, s in enumerate(ORDER):
    ax = fig.add_subplot(gs[0, i])
    x, y = MZ.loc[s].values, EZ.loc[s].values
    nc = len(CONT)
    lim = [min(x.min(), y.min()) - .25, max(x.max(), y.max()) + .25]
    ax.plot(lim, lim, color=MUT, linewidth=0.7, zorder=1)
    ax.scatter(x[:nc], y[:nc], s=15, color=EIC, alpha=.85, linewidth=0, zorder=2,
               label=T["f2_cont"])
    ax.scatter(x[nc:], y[nc:], s=20, facecolor="none", edgecolor=EIC, linewidth=.9,
               marker="s", zorder=3, label=T["f2_org"])
    r = np.corrcoef(x, y)[0, 1]
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(FLAT[s], fontsize=6.8, pad=4)
    ax.text(.04, .93, f"r = {r:.3f}", transform=ax.transAxes, fontsize=7.8,
            fontweight="bold", va="top", color=EIC)
    ax.tick_params(labelsize=6.6)
    if i == 0:
        ax.set_ylabel(T["f2_ylabel"], fontsize=7.2)
    ax.set_xlabel(T["f2_xlabel"], fontsize=7.2)
    if i == 3:
        h, lb = ax.get_legend_handles_labels()
        fig.legend(h, lb, fontsize=6.6, frameon=False, ncol=2, loc="upper center",
                   bbox_to_anchor=(.5, .478), handletextpad=.35, columnspacing=1.6)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

axb = fig.add_subplot(gs[1, :])
mp = (mw.state.value_counts(normalize=True) * 100).reindex(ORDER)
ep = (sub.state.value_counts(normalize=True) * 100).reindex(ORDER)
yy = np.arange(len(ORDER)); h = 0.36
axb.barh(yy + h/2, mp.values, height=h, color=MIM, label=T["f2_mimic"])
axb.barh(yy - h/2, ep.values, height=h, color=EIC, label=T["f2_eicu"])
for i, (m, e) in enumerate(zip(mp.values, ep.values)):
    axb.text(m + 1, i + h/2, f"{m:.1f}", va="center", fontsize=7.2, color=MIM)
    axb.text(e + 1, i - h/2, f"{e:.1f}", va="center", fontsize=7.2, color=EIC, fontweight="bold")
axb.set_yticks(yy); axb.set_yticklabels([FLAT[s] for s in ORDER], fontsize=7.4)
axb.invert_yaxis(); axb.set_xlabel(T["f2_pct"], fontsize=7.6)
axb.set_xlim(0, 78); axb.legend(fontsize=7.2, frameon=False, loc="lower right")
for sp in ["top", "right"]: axb.spines[sp].set_visible(False)
axb.tick_params(labelsize=6.8)
fig.savefig(FIG / f"figure2_transport_{LANG}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / f"figure2_transport_{LANG}.tif", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig); print(f"Figure 2 written ({LANG})")

# =====================================================================
# Figure 3 — de novo correspondence
# =====================================================================
def corrmat(model, n_states):
    Pm = np.array([[float(d.distributions[i].means.detach().ravel()[0]) for i in range(len(CONT))]
                   for d in mdl.distributions])
    Q = np.array([[float(d.distributions[i].means.detach().ravel()[0]) for i in range(len(CONT))]
                  for d in model.distributions])
    return np.array([[np.corrcoef(Q[j], Pm[k])[0, 1] for k in range(4)] for j in range(n_states)])

m4 = torch.load(HERE / "results_denovo/.model_k4_eicu_denovo.pt", weights_only=False)
m3 = torch.load(HERE / "results_denovo/.model_k3_eicu_denovo.pt", weights_only=False)
MIMIC_ORDER = [k for k in range(4)]
colnames = [FLAT[lblmap[k]] for k in MIMIC_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={"width_ratios": [1, 1.14], "wspace": .55})
for ax, model, ns, title in [(axes[0], m3, 3, T["f3_k3"]),
                             (axes[1], m4, 4, T["f3_k4"])]:
    C = corrmat(model, ns)
    im = ax.imshow(C, cmap="BuPu", vmin=0, vmax=1, aspect="auto")
    for i in range(ns):
        for j in range(4):
            best = C[i].argmax() == j
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if C[i, j] > .6 else INK,
                    fontweight="bold" if best else "normal")
            if best:
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor=EIC, linewidth=1.6))
    ax.set_xticks(range(4)); ax.set_xticklabels(colnames, rotation=30, ha="right", fontsize=6.3)
    if ns == 4:
        ax.add_patch(plt.Rectangle((.5, 1.5), 1, 2, fill=False, edgecolor="#97403A",
                                   linewidth=1.1, linestyle=(0, (3, 2)), zorder=5))
    ax.set_yticks(range(ns)); ax.set_yticklabels([T["f3_state"] + str(i) for i in range(ns)], fontsize=7)
    ax.set_title(title, fontsize=7.8, pad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=.024, pad=.05)
cb.set_label(T["f3_cbar"], fontsize=7.2)
cb.ax.tick_params(labelsize=6.6)
fig.subplots_adjust(bottom=.34)
fig.text(.5, .035, T["f3_note"],
         ha="center", fontsize=6.5, color="#444444", linespacing=1.6)
fig.savefig(FIG / f"figure3_denovo_correspondence_{LANG}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / f"figure3_denovo_correspondence_{LANG}.tif", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig); print(f"Figure 3 written ({LANG})")
print(f"\nAll figures -> {FIG}/")
