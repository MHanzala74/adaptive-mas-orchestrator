"""Figure 1 - system architecture, research-paper style (grayscale, serif, caption embedded).
Outputs docs/fig_architecture.png (300 dpi) and docs/fig_architecture.pdf (vector).
Run: python docs/make_figure.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})
fig = plt.figure(figsize=(10, 7.9))
ax = fig.add_axes([0.01, 0.135, 0.98, 0.86])
ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")
K, G = "black", "0.45"


def box(x, y, w, h, text, fs=9, fc="white", ls="-", lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc=fc, ec=K, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(p, q, ls="-", color=K, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def lab(x, y, t, fs=8, ha="center", style="italic"):
    ax.text(x, y, t, fontsize=fs, ha=ha, va="center", style=style)

# ---- input and router
box(1, 31, 13, 12, "Task $q$\n\nToken budget $B$")
box(21, 31, 17, 12, "Router\n(budget-aware LinUCB)\n\n$a^{*}=\\arg\\max_a\\,\\hat r_a(\\phi(q))$", fs=8.5)
arrow((14, 37), (21, 37)); lab(17.5, 39.5, "$\\phi(q)$", 8, style="normal")

# ---- architecture pool (five architectures)
ax.add_patch(FancyBboxPatch((45, 23), 34, 30, boxstyle="round,pad=0,rounding_size=0.8", fc="0.95", ec=K, lw=1.2))
ax.text(62, 51, "Architecture pool", ha="center", va="center", fontsize=9.5, fontweight="bold")
rows = [46.2, 41.2, 36.2, 31.2, 26.2]
names = ["(a) Single", "(b) Debate", "(c) Debate+Judge", "(d) Planner\u2013Worker", "(e) Solver\u2013Critic"]
for y, n in zip(rows, names):
    ax.text(46.3, y, n, fontsize=8, va="center", style="italic")
H = 3.4
box(62, rows[0] - H / 2, 8, H, "solo", fs=8.5)
for i, x in enumerate((62, 67.4, 72.8)):
    box(x, rows[1] - H / 2, 4.8, H, f"$D_{i}$", fs=8.5)
for i, x in enumerate((58.5, 63, 67.5)):
    box(x, rows[2] - H / 2, 3.9, H, f"$D_{i}$", fs=8.5)
box(72.5, rows[2] - H / 2, 4.6, H, "judge", fs=8, ls="--")
box(62, rows[3] - H / 2, 6.4, H, "planner", fs=8); box(70, rows[3] - H / 2, 6.4, H, "worker", fs=8)
arrow((68.4, rows[3]), (70, rows[3]))
box(62, rows[4] - H / 2, 6.4, H, "solver", fs=8); box(70, rows[4] - H / 2, 6.4, H, "critic", fs=8)
arrow((68.4, rows[4] + .5), (70, rows[4] + .5)); arrow((70, rows[4] - .5), (68.4, rows[4] - .5))
arrow((38, 37), (45, 37)); lab(41.5, 39.3, "$a^{*}$", 8.5, style="normal")

# ---- fault injector (evaluation only)
box(52, 58, 25, 5, "Fault injector (testbed)\nwrong \u00b7 malicious \u00b7 loop \u00b7 silent", fs=8, ls="--")
arrow((64.5, 58), (64.5, 53), ls="--", color=G)
lab(78, 60.5, "corrupts agent outputs\n(evaluation only)", 7.5, ha="left")

# ---- output
box(85, 31, 14, 12, "Answer $\\hat y$\n\nExecution\ntrace $\\tau$")
arrow((79, 37), (85, 37))

# ---- detect -> attribute -> repair
box(84, 8, 15, 12, "Failure detector\n(pass $\\Rightarrow$ return $\\hat y$)\nno answer \u00b7 degenerate\ndisagreement \u00b7 budget\nverifier mismatch", fs=8)
arrow((92, 31), (92, 20)); lab(95.7, 25.5, "$\\hat y,\\ \\tau$", 8.5, style="normal")
box(57, 8, 21, 12, "Failure attribution\n\nrule-based or LLM\n$\\Rightarrow$ culprit agent, step", fs=8.5)
box(29, 8, 21, 12, "Repair policy\n\nreprompt \u00b7 replace\nbypass \u00b7 switch topology", fs=8.5)
arrow((84, 14), (78, 14)); lab(81, 16.3, "fail", 8)
arrow((57, 14), (50, 14))
arrow((42, 20), (47, 25), rad=-0.35)
lab(36.5, 24.2, "overrides /\nnew topology", 7.5)

# ---- router update feedback
ax.plot([92, 92], [43, 66], color=G, lw=1.0, ls="--")
ax.plot([92, 29.5], [66, 66], color=G, lw=1.0, ls="--")
arrow((29.5, 66), (29.5, 43), ls="--", color=G)
lab(60.5, 67.3, "router update: $r=\\mathbb{1}[\\hat y=y]-\\lambda\\,\\mathrm{tokens}/300$", 8)

# ---- shared budget bar
box(1, 1.2, 98, 3.4, "All calls (initial run, verifier, attribution, retries) draw on one shared token budget $B$", fs=8.5, fc="0.9")
for x in (91.5, 67.5, 39.5):
    ax.plot([x, x], [4.6, 8], color=G, lw=0.8, ls=":")

# ---- embedded caption (paper style)
cap = ("$\\bf{Figure\\ 1.}$ Overview of the adaptive multi-agent orchestrator. A router maps a task $q$ and token budget $B$ to one of five architectures\n"
       "(a)\u2013(e). The answer $\\hat y$ and execution trace $\\tau$ are checked by a failure detector; on failure, attribution identifies the culprit agent and\n"
       "step, and the repair policy reprompts, replaces or bypasses it, or switches topology. Dashed elements: fault injector (evaluation only),\n"
       "router-update feedback, and the judge (called only when the vote has no majority). All model calls share one token budget.")
fig.text(0.015, 0.108, cap, ha="left", va="top", fontsize=9, linespacing=1.4)
fig.savefig("docs/fig_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("docs/fig_architecture.pdf", bbox_inches="tight", facecolor="white")
print("saved")
