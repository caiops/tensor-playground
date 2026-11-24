# Tensor Playground

An interactive teaching tool for learning what diffusion tensor imaging (DTI) metrics mean.
Change the properties of a simulated piece of white matter and see how the diffusion tensor
and the metrics FA, MD, AD and RD respond.

This is a **teaching model, not an analysis tool**. It is deliberately simplified, and its
parameters were chosen to produce plausible DTI values rather than calibrated against real
tissue. The tissue drawing is illustrative only.

**Try it online:** [link to the Streamlit app]

No installation or account is needed. It runs in any browser on a laptop, tablet or phone.

## What you can do

**Change the tissue** with five sliders:

| Slider | What it represents |
|---|---|
| Axon density | How much of the tissue is occupied by axons (lower = axonal loss) |
| Myelin | 1 = normal myelin, 0 = fully demyelinated |
| Axonal injury | Beading/swelling that slows diffusion along the axons |
| Free water | Fraction of the voxel filled with unrestricted water (CSF, oedema) |
| Fibre dispersion | How much fibre orientations spread around the main direction (not damage) |

**See the result:**
- the DTI metrics (FA, MD, AD, RD), compared with healthy tissue, and the eigenvalues λ₁-λ₃;
- the diffusion ellipsoid on a fixed scale;
- an illustrative drawing of the tissue, in cross-section and along the fibres.

Use **Reset to healthy** to return to the starting point at any time.

**Work through four challenges:**
1. **Grey matter:** build a voxel with FA < 0.20 and MD between 0.80 and 1.00.
2. **Same FA, different cause:** reach the same FA once through demyelination and once through axonal loss, and compare the metrics.
3. **Lower FA without damage:** reduce FA without changing axon density, myelin or axonal injury.
4. **What can you conclude?** Explain what a reduced FA does, and does not, tell you about the tissue.
