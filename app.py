import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from model import HEALTHY, run
from schematic import draw

st.set_page_config(page_title="Tensor Playground", page_icon="🧠", layout="wide")

UNIT = "x10⁻³ mm²/s"
PARAMS = ["axon_density", "myelin", "injury", "free_water", "dispersion"]
LABELS = {
    "axon_density": "Axon density",
    "myelin": "Myelin",
    "injury": "Axonal injury (beading)",
    "free_water": "Free water (oedema / CSF)",
    "dispersion": "Fibre dispersion (°)",
}

# -----
# state
# -----
for k, v in HEALTHY.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("done", set())
st.session_state.setdefault("c2_results", {})

def reset():
    for k, v in HEALTHY.items():
        st.session_state[k] = v

def changed():
    # parameters that differ from the healthy reference
    return {k for k in PARAMS if abs(st.session_state[k] - HEALTHY[k]) > 1e-6}

ref, _, _ = run(**HEALTHY)

# ------
# header
# ------
st.title("Tensor Playground")
st.caption("Change the tissue and watch how the diffusion tensor and the DTI metrics respond. "
           "Start from healthy white matter. Use **Reset to healthy** whenever you want to go back to the healthy state.")

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("Tissue")
    st.slider(LABELS["axon_density"], 0.2, 0.9, step=0.01, key="axon_density",
              help="Fraction of the tissue occupied by axons. Lower = axonal loss.")
    st.slider(LABELS["myelin"], 0.0, 1.0, step=0.01, key="myelin",
              help="1 = normal myelin; 0 = fully demyelinated. In the drawing, demyelination is patchy: the slider sets the fraction of axons that keep their sheath.")
    st.slider(LABELS["injury"], 0.0, 1.0, step=0.01, key="injury",
              help="Axonal beading/swelling that slows diffusion along the axons.")
    st.slider(LABELS["free_water"], 0.0, 1.0, step=0.01, key="free_water",
              help="Fraction of the voxel occupied by unrestricted water (CSF partial volume, oedema). Different from the space left between axons when they are lost, which is still hindered tissue water.")
    st.slider(LABELS["dispersion"], 0.0, 90.0, step=1.0, key="dispersion",
              help="How much fibre orientations spread around the main direction (fanning, crossing). Not damage.")
    st.button("Reset to healthy", on_click=reset, width="stretch")

p = {k: st.session_state[k] for k in PARAMS}
m, evals, evecs = run(**p)

with right:
    cards = "".join(
        f'<div style="flex:1;min-width:70px;padding:6px 8px;border:1px solid rgba(128,128,128,.3);border-radius:8px">'
        f'<div style="font-size:.8rem;opacity:.7">{k}</div>'
        f'<div style="font-size:1.6rem;font-weight:600;line-height:1.2">{m[k]:.2f}</div>'
        f'<div style="font-size:.72rem;opacity:.7">{m[k] - ref[k]:+.2f} vs healthy</div></div>'
        for k in ["FA", "MD", "AD", "RD"])
    st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:nowrap">{cards}</div>', unsafe_allow_html=True)
    st.caption(f"FA is dimensionless (0-1); MD, AD and RD in {UNIT}; eigenvalues ({UNIT}): λ₁ = {m['l1']:.2f} | λ₂ = {m['l2']:.2f} | λ₃ = {m['l3']:.2f}")

    g1, g2 = st.columns([1, 1.6])
    with g1:
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
        pts = np.einsum("ij,jkl->ikl", evecs @ np.diag(evals), sphere)
        fig = go.Figure(go.Surface(x=pts[0], y=pts[1], z=pts[2], surfacecolor=np.abs(pts[2]),
                                   colorscale="Blues", showscale=False, opacity=0.95,
                                   hoverinfo="skip",
                                   contours=dict(x=dict(highlight=False), y=dict(highlight=False),
                                                 z=dict(highlight=False))))
        rng = [-3.05, 3.05]   # fits the largest possible ellipsoid (free water, 3.0)
        fig.update_layout(
            height=330, margin=dict(l=0, r=0, t=30, b=0),
            uirevision="keep", hovermode=False,  # keep the zoom/rotation when sliders change
            title=dict(text="Diffusion ellipsoid", x=0.5, xanchor="center", xref="container", font=dict(size=14)),
            scene=dict(xaxis=dict(range=rng, visible=False, showspikes=False),
                       yaxis=dict(range=rng, visible=False, showspikes=False),
                       zaxis=dict(range=rng, visible=False, showspikes=False),
                       aspectmode="manual", aspectratio=dict(x=2.8, y=2.8, z=2.8),  # initial zoom: healthy tissue fills the view
                       dragmode="orbit",
                       camera=dict(eye=dict(x=1.0, y=1.0, z=0.4), projection=dict(type="orthographic"))),
        )
        with st.container(gap=None):
            with st.container(border=True):
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True})
            st.caption("Drag to rotate, scroll or pinch to zoom. The scale is fixed, so a bigger ellipsoid means more diffusion; zoom out to see very large ones (e.g., high free water).")
    with g2:
        tissue_fig = draw(**p)
        with st.container(gap=None):
            with st.container(border=True):
                st.pyplot(tissue_fig, width="stretch")
            plt.close(tissue_fig)
            st.caption("Illustrative drawing only.")

st.divider()

# ----------
# challenges
# ----------
st.subheader("Challenges")
t1, t2, t3, t4 = st.tabs(["1 · Grey matter", "2 · Same FA, different cause",
                          "3 · Lower FA without damage", "4 · What can you conclude?"])
ok = lambda txt: st.success("✓ " + txt)

with t1:
    st.markdown("Grey matter has many cell bodies and dendrites oriented in all directions. "
                "**Make a voxel with FA < 0.20 and MD between 0.80 and 1.00.**")
    fa_ok, md_ok = m["FA"] < 0.20, 0.80 <= m["MD"] <= 1.00
    if fa_ok and md_ok:
        st.session_state.done.add(1)
        ok(f"FA = {m['FA']:.2f}, MD = {m['MD']:.2f}. Which slider did most of the work? "
           "Low FA here comes from orientations spreading out, not from tissue damage.")
    elif not fa_ok:
        st.info(f"FA = {m['FA']:.2f} is still too high. Grey matter has no dominant direction: "
                "which slider controls how coherent the orientations are?")
    else:
        st.info(f"FA is low enough, but MD = {m['MD']:.2f} is outside 0.80-1.00. "
                + ("Too much free diffusion: grey matter is densely packed tissue, not fluid."
                   if m["MD"] > 1.0 else "Diffusion is too restricted; there is more extracellular space in grey matter."))

with t2:
    st.markdown("Start from **healthy** tissue. Reach **FA = 0.55 ± 0.02** in two different ways: "
                "(a) changing **only myelin**, then (b) **Reset** and change **only axon density**. "
                "Then compare all four metrics.")
    ch = changed()
    target = abs(m["FA"] - 0.55) <= 0.02
    for route, param in [("a", "myelin"), ("b", "axon_density")]:
        if ch == {param} and target:
            st.session_state.c2_results[route] = dict(m)
    res = st.session_state.c2_results
    if ch and ch not in ({"myelin"}, {"axon_density"}):
        st.warning("For this challenge change only one slider: "
                   + ", ".join(LABELS[k] for k in sorted(ch)) + " changed. Press **Reset to healthy**.")
    elif ch and target and not {"a", "b"} <= res.keys():
        st.info("Target reached. Now press **Reset to healthy** and try the other route.")
    elif ch and not target:
        st.info(f"FA = {m['FA']:.2f}. " + ("Keep going." if m["FA"] > 0.57 else "You went past the target; move back a little."))
    st.write(f"(a) Demyelination: {'✓ done' if 'a' in res else '—'}    (b) Axon loss: {'✓ done' if 'b' in res else '—'}")
    if {"a", "b"} <= res.keys():
        st.session_state.done.add(2)
        rows = {"Healthy": ref, "(a) Demyelination": res["a"], "(b) Axon loss": res["b"]}
        st.table({name: {k: f"{r[k]:.2f}" for k in ["FA", "MD", "AD", "RD"]} for name, r in rows.items()})
        ok("Two different pathologies, practically the same FA, MD, AD and RD. "
           "Could a DTI study tell them apart? What would you need to measure instead?")

with t3:
    st.markdown("Start from **healthy** tissue. **Lower FA by at least 0.10 without touching axon density, "
                "myelin or axonal injury.**")
    ch = changed()
    damage = ch & {"axon_density", "myelin", "injury"}
    drop = ref["FA"] - m["FA"]
    if damage:
        st.warning("That counts as damage: " + ", ".join(LABELS[k] for k in sorted(damage))
                   + " changed. Press **Reset to healthy** and try the other sliders.")
    elif drop >= 0.10:
        st.session_state.done.add(3)
        used = " and ".join(LABELS[k] for k in sorted(ch))
        ok(f"FA dropped by {drop:.2f} using {used}, with perfectly healthy axons. "
           "Compare how MD and AD changed here with challenge 2: free water raises MD, "
           "dispersion lowers AD while MD barely moves.")
    else:
        st.info(f"FA has dropped by {max(drop, 0):.2f} so far.")

with t4:
    st.markdown("Discuss with your partner and write one or two sentences: **a study reports lower FA in a "
                "patient group. What can, and can't, you conclude about the tissue?**")
    ans = st.text_area("Your answer", key="reflection", height=90,
                       help="Kept while this page is open; refreshing the page clears it.")
    if ans.strip():
        st.markdown("At the end of the session, paste this answer into the **exit questionnaire** "
                    "(use the copy icon at the top right of the box below).")
        st.code(ans, language=None, wrap_lines=True)

with st.expander("About this model"):
    st.markdown(f"""
This is a **teaching model**, not an analysis tool. Each voxel contains three compartments:
water inside axons (diffusion only along the axon), water around axons (hindered: radial diffusion falls as
axons pack more densely and myelin is intact), and free water (D = 3.0 {UNIT}). Axonal injury slows
diffusion along the axons; demyelination lets water move radially. In the drawing, demyelination is shown as
patchy loss (some axons lose their sheath entirely); in the model it is applied as an average over the voxel,
which gives the same DTI metrics. Fibre orientations spread according to the dispersion slider. The model
simulates a noise-free acquisition (b = 1000 s/mm², 30 directions) and fits a diffusion tensor as DTI software
would. Losing axons enlarges the space between them, but that water is still hindered by the remaining tissue,
so it is not the same as free water; in disease the two often increase together (e.g., atrophy with CSF expansion),
but the model keeps them as separate sliders so their effects can be compared. Parameter values are chosen to give
plausible numbers, not calibrated against real tissue.
""")