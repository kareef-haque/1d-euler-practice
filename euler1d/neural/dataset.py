'''
Training / evaluation data for the neural flux.

Two ground-truth sources:

1. 'exact' (default, the NFV approach): exact Riemann solutions averaged over each cell,
   the analogue of NFV's Lax-Hopf ground truth. States are drawn from the same
   distributions as infrastructure/physics_config.py (rho~U(0.1,2.5), u~N(0,5),
   P~U(1e4,1.25e5), interface in [20%, 80%] of the window). Ghost cells come from the exact
   solution too, so the learned scheme is never trained against boundary artefacts
   (NFV pins its boundary cells to ground truth in the same way).

2. 'numerical': coarse-grained trajectories written by data_generation.py
   (conservative cell averaging in space, striding in time). Accuracy is capped by the
   fine reference scheme itself.

Returned arrays:  Q (B, T, 3, N)  and  ghosts (B, T, 3, 2*n_ghost) or None.
'''
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

from infrastructure.results import ExactRiemannSolver


@dataclass
class RiemannSpec:
    prim_L: tuple   # (rho, u, P)
    prim_R: tuple
    x_frac: float   # interface position as a fraction of the window length


def sample_riemann_specs(n, seed, rho_range=(0.1, 2.5), u_std=5.0, p_range=(1e4, 1.25e5),
                         x_frac_range=(0.2, 0.8)):
    '''Same distributions as PhysicsConfig, but with an explicit seed (train/eval separation).
    The interface is continuous in [x_frac_range], so it falls at arbitrary sub-cell positions
    (plays the role of NFV's --x_noise).'''
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(n):
        rL, rR = rng.uniform(*rho_range, size=2)
        uL, uR = rng.normal(0.0, u_std, size=2)
        pL, pR = rng.uniform(*p_range, size=2)
        specs.append(RiemannSpec((rL, uL, pL), (rR, uR, pR), float(rng.uniform(*x_frac_range))))
    return specs


def sod_spec(x_frac=0.5):
    return RiemannSpec((1.0, 0.0, 1e5), (0.125, 0.0, 1e4), x_frac)


def sample_exact(ers, x, t):
    '''Vectorised equivalent of ExactRiemannSolver.sample (same branch logic; checked in tests).'''
    g = ers.gamma
    if t <= 1e-12:
        left = x < ers.x_int
        return (np.where(left, ers.rho_L, ers.rho_R), np.where(left, ers.u_L, ers.u_R),
                np.where(left, ers.P_L, ers.P_R))
    S = (x - ers.x_int) / t
    rho, u, p = np.empty_like(S), np.empty_like(S), np.empty_like(S)

    def put(mask, r, v, q):
        rho[mask], u[mask], p[mask] = (r[mask] if np.ndim(r) else r,
                                       v[mask] if np.ndim(v) else v,
                                       q[mask] if np.ndim(q) else q)

    left = S < ers.u_star
    right = ~left
    if ers.P_star > ers.P_L:
        SL = ers.u_L - ers.a_L * np.sqrt((g + 1) / (2 * g) * (ers.P_star / ers.P_L) + (g - 1) / (2 * g))
        put(left & (S < SL), ers.rho_L, ers.u_L, ers.P_L)
        put(left & (S >= SL), ers.rho_star_L, ers.u_star, ers.P_star)
    else:
        SH = ers.u_L - ers.a_L
        ST = ers.u_star - ers.a_L * (ers.P_star / ers.P_L) ** ((g - 1) / (2 * g))
        put(left & (S < SH), ers.rho_L, ers.u_L, ers.P_L)
        put(left & (S >= SH) & (S > ST), ers.rho_star_L, ers.u_star, ers.P_star)
        fan = left & (S >= SH) & (S <= ST)
        uf = 2 / (g + 1) * (ers.a_L + (g - 1) / 2 * ers.u_L + S)
        af = np.maximum(2 / (g + 1) * (ers.a_L + (g - 1) / 2 * (ers.u_L - S)), 0.0)  # clip: only used inside fan
        put(fan, ers.rho_L * (af / ers.a_L) ** (2 / (g - 1)), uf, ers.P_L * (af / ers.a_L) ** (2 * g / (g - 1)))
    if ers.P_star > ers.P_R:
        SR = ers.u_R + ers.a_R * np.sqrt((g + 1) / (2 * g) * (ers.P_star / ers.P_R) + (g - 1) / (2 * g))
        put(right & (S > SR), ers.rho_R, ers.u_R, ers.P_R)
        put(right & (S <= SR), ers.rho_star_R, ers.u_star, ers.P_star)
    else:
        SH = ers.u_R + ers.a_R
        ST = ers.u_star + ers.a_R * (ers.P_star / ers.P_R) ** ((g - 1) / (2 * g))
        put(right & (S > SH), ers.rho_R, ers.u_R, ers.P_R)
        put(right & (S <= SH) & (S < ST), ers.rho_star_R, ers.u_star, ers.P_star)
        fan = right & (S <= SH) & (S >= ST)
        uf = 2 / (g + 1) * (-ers.a_R + (g - 1) / 2 * ers.u_R + S)
        af = np.maximum(2 / (g + 1) * (ers.a_R - (g - 1) / 2 * (ers.u_R - S)), 0.0)
        put(fan, ers.rho_R * (af / ers.a_R) ** (2 / (g - 1)), uf, ers.P_R * (af / ers.a_R) ** (2 * g / (g - 1)))
    return rho, u, p


def exact_cell_averages(spec, nx, nt, dx, dt, gamma=1.4, n_ghost=3, subsamples=8):
    '''Cell averages of the conservative variables, (nt, 3, nx + 2*n_ghost), times t = k*dt.'''
    L = nx * dx
    ers = ExactRiemannSolver(spec.prim_L, spec.prim_R, x_interface=spec.x_frac * L, gamma=gamma)
    n_all = nx + 2 * n_ghost
    left_edges = (np.arange(n_all) - n_ghost) * dx
    offs = (np.arange(subsamples) + 0.5) / subsamples * dx
    x = (left_edges[:, None] + offs[None, :]).ravel()
    out = np.empty((nt, 3, n_all))
    for k in range(nt):
        rho, u, p = sample_exact(ers, x, k * dt)
        Q = np.stack([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u ** 2])     # average CONSERVED vars
        out[k] = Q.reshape(3, n_all, subsamples).mean(-1)
    return out


def build_exact_dataset(specs, nx, nt, dx, dt, gamma=1.4, n_ghost=3, subsamples=8,
                        dtype=torch.float64, device='cpu'):
    full = np.stack([exact_cell_averages(s, nx, nt, dx, dt, gamma, n_ghost, subsamples) for s in specs])
    full = torch.as_tensor(full, dtype=dtype, device=device)
    g = n_ghost
    Q = full[..., g:-g].contiguous()
    ghosts = torch.cat([full[..., :g], full[..., -g:]], dim=-1).contiguous()
    return Q, ghosts


# ------------------------------------------------------------------ numerical reference data
def load_numerical_dataset(folder, coarsen, time_stride, max_steps=None, max_files=None,
                           dtype=torch.float64, device='cpu'):
    '''
    Load *.npy trajectories (T, 3, N_fine) from data_generation.py and coarse-grain them.
    If a matching *_meta.npz exists (written by the patched data_generation.py), only the
    uniformly spaced part of t_hist is kept. Returns Q (B,T,3,N), ghosts=None, meta dict.
    '''
    files = sorted(f for f in glob.glob(os.path.join(folder, '*.npy')))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f'no .npy trajectories in {folder}')
    trajs, dx_f, dt_f = [], None, None
    for f in files:
        Q = np.load(f)
        meta_path = f[:-4] + '_meta.npz'
        if os.path.exists(meta_path):
            m = np.load(meta_path)
            dx_f, dt_f = float(m['dx']), float(m['dt'])
            dts = np.diff(m['t_hist'])
            bad = ~np.isclose(dts, dt_f, rtol=1e-6)
            n_ok = int(np.argmax(bad)) if bad.any() else len(dts)
            Q = Q[:n_ok + 1]
        T, _, Nf = Q.shape
        if Nf % coarsen:
            raise ValueError(f'N_fine={Nf} not divisible by coarsen={coarsen}')
        Qc = Q.reshape(T, 3, Nf // coarsen, coarsen).mean(-1)[::time_stride]
        trajs.append(Qc)
    T = min(len(q) for q in trajs)
    if max_steps:
        T = min(T, max_steps + 1)
    data = torch.as_tensor(np.stack([q[:T] for q in trajs]), dtype=dtype, device=device)
    meta = {'dx': dx_f * coarsen if dx_f else None, 'dt': dt_f * time_stride if dt_f else None}
    return data, None, meta


def state_scales(Q0, gamma=1.4):
    '''Per-sample channel scales [rho_ref, rho_ref*c_ref, p_ref] -> (B,3,1), used to normalise losses.'''
    rho = Q0[:, 0]
    u = Q0[:, 1] / rho
    p = (gamma - 1) * (Q0[:, 2] - 0.5 * rho * u ** 2)
    r, pr = rho.mean(-1), p.mean(-1)
    c = torch.sqrt(gamma * pr / r)
    return torch.stack([r, r * c, pr], dim=1).unsqueeze(-1)


def max_wave_speed(Q, gamma=1.4):
    rho = Q[..., 0, :]
    u = Q[..., 1, :] / rho
    p = (gamma - 1) * (Q[..., 2, :] - 0.5 * rho * u ** 2)
    return float((u.abs() + torch.sqrt(gamma * p.clamp_min(1e-12) / rho)).max())
