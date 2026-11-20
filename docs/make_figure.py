"""Fig. 1 - system architecture, journal style (grayscale, serif, vector PDF + 300 dpi PNG).
Run: python docs/make_figure.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})
fig, ax = plt.subplots(figsize=(10, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
K, G = "black", "0.45"


def box(x, y, w, h, text, fs=9, fc="white", ls="-", lw=1.0, bold=False, round_=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=fc, ec=K, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(p, q, ls="-", color=K, lw=1.0, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def lab(x, y, t, fs=8, ha="center", style="italic"):
    ax.text(x, y, t, fontsize=fs, ha=ha, va="center", style=style)

# ---- (1) input and router
box(1, 33, 13, 12, "Task $q$\n\nToken budget $B$")
box(21, 33, 17, 12, "Router\n(budget-aware LinUCB)\n\n$a^{*}=\\arg\\max_a\\,\\hat r_a(\\phi(q))$", fs=8.5)
arrow((14, 39), (21, 39))
lab(17.5, 41.5, "$\\phi(q)$", 8, style="normal")

# ---- (2) architecture pool
ax.add_patch(FancyBboxPatch((45, 24), 31, 27, boxstyle="round,pad=0,rounding_size=0.8", fc="0.95", ec=K, lw=1.2))
ax.text(60.5, 49, "Architecture pool", ha="center", va="center", fontsize=9.5, fontweight="bold")
# single
ax.text(46.5, 45, "(a) Single", fontsize=8, va="center", style="italic")
box(62, 43.2, 12, 3.6, "solo", fs=8.5)
# debate
ax.text(46.5, 38.6, "(b) Debate", fontsize=8, va="center", style="italic")
ax.text(64.4, 35.3, "majority vote over $D_0$\u2013$D_2$ (2 rounds)", fontsize=7.3, ha="center", va="center", style="italic")
for i, x in enumerate((55, 61.3, 67.6)):
    box(x, 36.8, 5.6, 3.6, f"$D_{i}$", fs=8.5)
# planner-worker
ax.text(46.5, 31.3, "(c) Planner\u2013Worker", fontsize=8, va="center", style="italic")
box(61, 29.4, 6.2, 3.6, "planner", fs=8)
box(68.8, 29.4, 6.2, 3.6, "worker", fs=8)
arrow((67.2, 31.2), (68.8, 31.2))
# shared note
ax.text(60.5, 26.2, "agents: $\\{$solo, $D_0$\u2013$D_2$, planner, worker$\\}$", fontsize=7.5, ha="center", va="center")
arrow((38, 39), (45, 39))
lab(41.5, 41.3, "$a^{*}$", 8.5, style="normal")

# ---- fault injector (evaluation only)
box(48, 55.5, 25, 5, "Fault injector (testbed)\nwrong \u00b7 malicious \u00b7 loop \u00b7 silent", fs=8, ls="--", fc="white")
arrow((60.5, 55.5), (60.5, 51), ls="--", color=G)
lab(75.2, 53.2, "corrupts agent outputs\n(evaluation only)", 7.5, ha="left")

# ---- output
box(83, 33, 15, 12, "Answer $\\hat y$\n\nExecution\ntrace $\\tau$")
arrow((76, 39), (83, 39))

# ---- (3) detect -> attribute -> repair loop
box(83, 9, 15, 12, "Failure detector\n(pass $\\Rightarrow$ return $\\hat y$)\nno answer \u00b7 degenerate\ndisagreement \u00b7 budget\nverifier mismatch", fs=8)
arrow((90.5, 33), (90.5, 21))
box(55, 9, 22, 12, "Failure attribution\n\nrule-based or LLM\n$\\Rightarrow$ culprit agent, step", fs=8.5)
box(27, 9, 22, 12, "Repair policy\n\nreprompt \u00b7 replace\nbypass \u00b7 switch topology", fs=8.5)
arrow((83, 15), (77, 15)); lab(80, 17.2, "fail", 8)
arrow((55, 15), (49, 15))
arrow((40, 21), (45, 27), rad=-0.35)
lab(34.5, 25.2, "overrides /\nnew topology", 7.5, ha="center")
lab(94.6, 27, "$\\hat y,\\ \\tau$", 8.5, style="normal")

# ---- router update (feedback, dashed)
ax.plot([90.5, 90.5], [45, 63], color=G, lw=1.0, ls="--")
ax.plot([90.5, 29.5], [63, 63], color=G, lw=1.0, ls="--")
arrow((29.5, 63), (29.5, 45), ls="--", color=G)
lab(60, 64.6, "router update: $r=\\mathbb{1}[\\hat y=y]-\\lambda\\,\\mathrm{tokens}/300$", 8)

# ---- shared budget bar
box(1, 1.5, 97, 3.6, "All calls (initial run, verifier, attribution, retries) draw on one shared token budget $B$", fs=8.5, fc="0.9")
for x in (90.5, 66, 38):
    ax.plot([x, x], [5.1, 9], color=G, lw=0.8, ls=":")

fig.savefig("docs/fig_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("docs/fig_architecture.pdf", bbox_inches="tight", facecolor="white")
print("saved")
