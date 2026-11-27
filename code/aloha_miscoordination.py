"""
Miscoordination in the two-node saturated slotted ALOHA game.

Shared reward (total throughput, successful packets per slot):
    S(p1, p2) = p1 (1 - p2) + p2 (1 - p1) = p1 + p2 - 2 p1 p2

Each node transmits in a slot with probability p_i; a packet succeeds iff
the other node stays silent.

The figure shows the defining feature of miscoordination: each agent's
update, taken alone, RAISES the joint reward; the two updates taken
together DESTROY it.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def S(p1, p2):
    return p1 + p2 - 2.0 * p1 * p2


# ----------------------------------------------------------------------
# The four joint policies being compared
# ----------------------------------------------------------------------

START = (0.25, 0.25)
A1_ONLY = (1.00, 0.25)      # agent 1 best-responds, agent 2 stands still
A2_ONLY = (0.25, 1.00)      # agent 2 best-responds, agent 1 stands still
BOTH = (1.00, 1.00)         # both best-respond at the same time

cases = [
    ("no update",    START,   S(*START)),
    ("agent 1 only", A1_ONLY, S(*A1_ONLY)),
    ("agent 2 only", A2_ONLY, S(*A2_ONLY)),
    ("both at once", BOTH,    S(*BOTH)),
]

print("=" * 58)
print("TWO-NODE SLOTTED ALOHA:  S(p1,p2) = p1 + p2 - 2 p1 p2")
print("=" * 58)
for name, p, v in cases:
    print(f"  {name:<14} p = ({p[0]:.2f}, {p[1]:.2f})    S = {v:.3f}")
print()
print(f"  each agent alone improves on {S(*START):.3f}")
print(f"  together they reach {S(*BOTH):.3f}  <- every slot collides")

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# Okabe-Ito, CVD-safe. Identity is never carried by colour alone: every
# bar is directly labelled and every arrow endpoint is annotated.
BLUE = "#0072B2"    # an agent improving things on its own
VERM = "#D55E00"    # the two of them together
INK = "#1A1A1A"
MUTED = "#6E6E6E"

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.12, 1.0]}
)

# ---------------- left: what each agent decides to do -----------------

g = np.linspace(0, 1, 400)
P1, P2 = np.meshgrid(g, g)
cf = axL.contourf(P1, P2, S(P1, P2), levels=np.linspace(0, 1, 21),
                  cmap="Greys", vmin=-0.10, vmax=2.3)
axL.contour(P1, P2, S(P1, P2), levels=np.linspace(0.2, 0.8, 4),
            colors=MUTED, linewidths=0.35, alpha=0.55)

arrow = dict(arrowstyle="-|>", lw=1.1, color=BLUE, shrinkA=4, shrinkB=2)
axL.annotate("", xy=A1_ONLY, xytext=START, arrowprops=arrow)
axL.annotate("", xy=A2_ONLY, xytext=START, arrowprops=arrow)
axL.annotate("", xy=BOTH, xytext=START,
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=VERM,
                             shrinkA=4, shrinkB=2))

halo = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
for (p, txt, col, at, ha, va) in [
    (A1_ONLY, r"$S=0.75$", BLUE, (0.97, 0.35), "right", "bottom"),
    (A2_ONLY, r"$S=0.75$", BLUE, (0.38, 0.97), "left", "top"),
    (BOTH, r"$S=0$", VERM, (0.90, 0.84), "right", "top"),
    (START, r"$S=0.375$", MUTED, (0.30, 0.17), "left", "top"),
]:
    axL.plot(*p, "o", ms=6.5, color=col, mec="white", mew=1.1, zorder=6)
    axL.annotate(txt, xy=p, xytext=at, ha=ha, va=va,
                 fontsize=8.5, color=col, weight="bold", bbox=halo, zorder=7)

axL.set_xlabel(r"agent 1 transmission probability $p_1$")
axL.set_ylabel(r"agent 2 transmission probability $p_2$")
axL.set_xlim(-0.04, 1.08)
axL.set_ylim(-0.04, 1.08)
axL.set_aspect("equal")
axL.set_xticks([0, 0.5, 1.0])
axL.set_yticks([0, 0.5, 1.0])
axL.tick_params(labelsize=8)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)
axL.set_title("each agent's update", fontsize=9.5, pad=7)

cbar = fig.colorbar(cf, ax=axL, ticks=[0, 0.5, 1.0], pad=0.03, fraction=0.045)
cbar.set_label(r"throughput $S$", fontsize=8)
cbar.ax.tick_params(labelsize=7.5, width=0.6)
cbar.outline.set_linewidth(0.6)

# ---------------- right: what it costs --------------------------------

names = ["no\nupdate", "agent 1\nonly", "agent 2\nonly", "both\nat once"]
vals = [c[2] for c in cases]
cols = [MUTED, BLUE, BLUE, VERM]
x = np.arange(4)

axR.bar(x, vals, width=0.6, color=cols, zorder=3)
# A zero-height bar is invisible, so mark the baseline explicitly.
axR.bar([3], [0.012], width=0.6, color=VERM, zorder=3)

axR.axhline(S(*START), color=INK, lw=0.7, ls=":", zorder=2)
axR.annotate("starting throughput", xy=(2.46, S(*START)),
             xytext=(2.46, S(*START) + 0.015),
             fontsize=7.5, color=INK, ha="right", va="bottom")

for xi, v in zip(x, vals):
    axR.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 4),
                 textcoords="offset points", ha="center",
                 fontsize=9, weight="bold",
                 color=VERM if v == 0 else INK)

axR.annotate("every slot\ncollides", xy=(3, 0.015), xytext=(3, 0.235),
             fontsize=8, color=VERM, ha="center", va="bottom", style="italic",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=VERM,
                             shrinkA=3, shrinkB=3))

axR.set_xticks(x)
axR.set_xticklabels(names, fontsize=8.5)
axR.set_ylabel(r"total throughput $S$")
axR.set_ylim(0, 0.92)
axR.set_yticks([0, 0.25, 0.5, 0.75])
axR.tick_params(labelsize=8)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)
axR.set_title("what the joint policy achieves", fontsize=9.5, pad=7)

fig.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(f"latex/figures/code/aloha_miscoordination.{ext}", bbox_inches="tight", dpi=200)
print("\nwrote aloha_miscoordination.pdf and .png")