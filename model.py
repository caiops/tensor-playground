import numpy as np

# Toy multi-compartment white-matter model for teaching DTI.

# Tissue parameters -> diffusion-weighted signals (b = 1000 s/mm^2, 30 directions)
# -> log-linear tensor fit -> eigenvalues and DTI metrics.

# Simplified on purpose: noise-free, single b-shell, no exchange between compartments,
# and parameter values chosen to give plausible DTI values rather than calibrated against real tissue.
# Diffusivities are in units of 1e-3 mm^2/s (um^2/ms).

B_VALUE = 1.0                   # 1000 s/mm^2 expressed in ms/um^2 (so b*D is dimensionless)
D_AXIAL = 1.7                   # intrinsic diffusivity along axons (intra and extra)
D_FREE = 3.0                    # free water at body temperature
INTRA_PERP_DEMYELINATED = 0.5   # radial leakage of intra-axonal water with no myelin
INJURY_AXIAL_DROP = 0.65        # fraction of intra-axonal D_axial lost at full injury

HEALTHY = dict(axon_density=0.6, myelin=1.0, injury=0.0, free_water=0.0, dispersion=25.0)

def _fibonacci_hemisphere(n):
    i = np.arange(n) + 0.5
    z = 1 - i / n                      # z in (0, 1]: one hemisphere is enough for DTI
    r = np.sqrt(1 - z ** 2)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

GRADIENTS = _fibonacci_hemisphere(30)

# fixed random draws so the fibre orientations do not jitter between reruns
_rng = np.random.default_rng(7)
_N_FIBRES = 400
_THETA_UNIT = np.abs(_rng.standard_normal(_N_FIBRES))   # polar angle in units of sigma
_PHI = _rng.uniform(0, 2 * np.pi, _N_FIBRES)

def fibre_orientations(dispersion_deg):
    # unit vectors around z with Gaussian-like polar spread (dispersion in degrees)
    theta = np.deg2rad(dispersion_deg) * _THETA_UNIT
    theta = np.minimum(theta, np.pi / 2)
    return np.stack([np.sin(theta) * np.cos(_PHI), np.sin(theta) * np.sin(_PHI), np.cos(theta)], axis=1)

def compartment_diffusivities(axon_density, myelin, injury):
    # return (intra_par, intra_perp, extra_par, extra_perp)
    intra_par = D_AXIAL * (1 - INJURY_AXIAL_DROP * injury)
    intra_perp = INTRA_PERP_DEMYELINATED * (1 - myelin)
    extra_par = D_AXIAL
    # tortuosity approximation: extra-axonal radial diffusivity falls as axons pack more densely;
    # myelin loss weakens the hindrance.
    hindrance = axon_density * (0.4 + 0.6 * myelin)
    extra_perp = extra_par * (1 - hindrance)
    return intra_par, intra_perp, extra_par, extra_perp

def simulate_signal(axon_density, myelin, injury, free_water, dispersion):
    g = GRADIENTS
    n = fibre_orientations(dispersion)
    cos2 = (g @ n.T) ** 2                      # (n_grad, n_fibres)
    ip, iperp, ep, eperp = compartment_diffusivities(axon_density, myelin, injury)
    s_intra = np.exp(-B_VALUE * (ip * cos2 + iperp * (1 - cos2))).mean(axis=1)
    s_extra = np.exp(-B_VALUE * (ep * cos2 + eperp * (1 - cos2))).mean(axis=1)
    s_tissue = axon_density * s_intra + (1 - axon_density) * s_extra
    s_free = np.exp(-B_VALUE * D_FREE)
    return (1 - free_water) * s_tissue + free_water * s_free

def fit_tensor(signal):
    g = GRADIENTS
    X = np.stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                  2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]], axis=1)
    y = -np.log(signal) / B_VALUE
    dxx, dyy, dzz, dxy, dxz, dyz = np.linalg.lstsq(X, y, rcond=None)[0]
    D = np.array([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
    evals, evecs = np.linalg.eigh(D)
    order = np.argsort(evals)[::-1]
    return np.clip(evals[order], 1e-6, None), evecs[:, order]

def dti_metrics(evals):
    l1, l2, l3 = evals
    md = evals.mean()
    fa = np.sqrt(1.5) * np.linalg.norm(evals - md) / np.linalg.norm(evals)
    return dict(FA=float(fa), MD=float(md), AD=float(l1), RD=float((l2 + l3) / 2),
                l1=float(l1), l2=float(l2), l3=float(l3))

def run(axon_density, myelin, injury, free_water, dispersion):
    sig = simulate_signal(axon_density, myelin, injury, free_water, dispersion)
    evals, evecs = fit_tensor(sig)
    return dti_metrics(evals), evals, evecs