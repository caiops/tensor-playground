import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

BOX = 10.0
_rng = np.random.default_rng(3)
MYELIN_MAX = 0.38      # myelin ring adds up to 38% to the axon radius
MIN_GAP = 0.12         # minimum space between neighbouring myelin sheaths (box units)

def _pack(n_try=20000, r_min=0.28, r_max=0.48):
    # random non-overlapping axons, spaced using their outer radius at full myelin
    pts = []
    for _ in range(n_try):
        r = _rng.uniform(r_min, r_max)
        R = r * (1 + MYELIN_MAX)
        x, y = _rng.uniform(R + 0.05, BOX - R - 0.05, 2)
        if all((x - px) ** 2 + (y - py) ** 2 > (R + pr * (1 + MYELIN_MAX) + MIN_GAP) ** 2 for px, py, pr in pts):
            pts.append((x, y, r))
    order = _rng.permutation(len(pts))
    return [pts[i] for i in order]

AXON_POOL = _pack()
# an axon keeps its sheath when its draw is below the myelin slider, so demyelination is patchy
# (some axons bare, others intact) rather than uniform thinning.
MYELIN_DRAWS = _rng.uniform(0, 1, len(AXON_POOL))
FIBRE_DRAWS = _rng.standard_normal(14)
FIBRE_X = np.linspace(0.8, BOX - 0.8, 14)
FIBRE_MYELIN_DRAWS = _rng.uniform(0, 1, 14)

WATER = "#a9cdec"
WATER_EDGE = "#3b82c4"
MYELIN = "#c9a227"
AXON = "#f4efe6"
INJURED = "#d98c6f"
TISSUE_BG = "#efe9df"
AXOLEMMA = "#8c8273"

def draw(axon_density, myelin, injury, free_water, dispersion):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=110)
    for ax in (ax1, ax2):
        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(TISSUE_BG)
        for s in ax.spines.values():
            s.set_color("#999")

    # free water is drawn as a separate CSF-like pool occupying that fraction of the voxel (partial volume)
    # the beige space between axons is hindered extra-axonal water
    edge = (1 - free_water) * BOX
    axon_fill = matplotlib.colors.to_hex(
        (1 - injury) * np.array(matplotlib.colors.to_rgb(AXON)) + injury * np.array(matplotlib.colors.to_rgb(INJURED)))

    def pool(ax):
        if free_water > 0.005:
            ax.add_patch(Rectangle((edge, 0), BOX - edge, BOX, facecolor=WATER, edgecolor=WATER_EDGE,
                                   lw=1.0, zorder=5))
            if free_water > 0.12:
                ax.text(edge + (BOX - edge) / 2, BOX / 2, "free\nwater", ha="center", va="center",
                        fontsize=8, color="#1f4e79", zorder=6)

    # cross-section: number of axons follows density, ring thickness follows myelin
    n_ax = int(round(axon_density * len(AXON_POOL)))
    for i, (x, y, r) in enumerate(AXON_POOL[:n_ax]):
        ring = r * (1 + MYELIN_MAX)
        if x + ring > edge:
            continue
        if MYELIN_DRAWS[i] < myelin:
            ax1.add_patch(Circle((x, y), ring, color=MYELIN, lw=0))
        ax1.add_patch(Circle((x, y), r, facecolor=axon_fill, edgecolor=AXOLEMMA, lw=0.8))
    pool(ax1)
    ax1.set_title("Cross-section", fontsize=12)

    # longitudinal: orientation spread follows dispersion, beads follow injury
    n_fib = int(round(axon_density * len(FIBRE_X)))
    idx = np.linspace(0, len(FIBRE_X) - 1, n_fib).astype(int) if n_fib else []
    y = np.linspace(-1, BOX + 1, 60)
    for k in idx:
        ang = np.deg2rad(dispersion) * FIBRE_DRAWS[k] * 0.45
        x = FIBRE_X[k] + np.tan(np.clip(ang, -1.2, 1.2)) * (y - BOX / 2)
        if FIBRE_MYELIN_DRAWS[k] < myelin:
            ax2.plot(x, y, color=MYELIN, lw=6.0, solid_capstyle="butt", zorder=1)
        ax2.plot(x, y, color=AXOLEMMA, lw=2.6, zorder=2)
        ax2.plot(x, y, color=AXON, lw=1.6, zorder=2)
        if injury > 0.05:
            yb = y[5::9]
            xb = FIBRE_X[k] + np.tan(np.clip(ang, -1.2, 1.2)) * (yb - BOX / 2)
            ax2.scatter(xb, yb, s=8 + 70 * injury, color=INJURED, edgecolor="#8a4a33", lw=0.5, zorder=3)
    pool(ax2)
    ax2.set_title("Along the fibres", fontsize=12)

    fig.text(0.5, 0.015, "gold = myelin sheath  ·  light = axon  ·  beige = water between axons (hindered)",
             ha="center", fontsize=11, color="#555")
    fig.text(0.5, -0.04, "orange = injury  ·  blue = free water",
             ha="center", fontsize=11, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig