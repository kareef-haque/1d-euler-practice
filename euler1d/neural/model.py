'''
Neural numerical flux for the 1D Euler equations (NFV adapted to a hyperbolic *system*).

NFV (scalar LWR) learns F(k_i, k_{i+1}) with a Conv1d(kernel=2) + 1x1-conv MLP and clamps
the output to [0, q_max]. For Euler that clamp is invalid (mass/energy fluxes change sign) and
raw flux regression is poorly conditioned (rho ~ 1, p ~ 1e5). This model instead learns a
bounded, dimensionless correction on top of a classical flux:

    F_{j+1/2} = F_base(QL, QR) + D_j * G_sym(z_j)

  z_j      dimensionless stencil features  [log(rho/rho_ref), u/c_ref, log(p/p_ref)] over S cells,
           with rho_ref, p_ref from the two cells adjacent to the interface
           (Euler is invariant under rho -> a*rho, p -> b*p, u -> sqrt(b/a)*u, so the network
           only ever sees O(1) inputs and generalises across that scaling family)
  D_j      flux scales [rho_ref*c_ref, p_ref, p_ref*c_ref]
  G(z)     = delta(z) * g_max * tanh(MLP(z)),  delta = d/(1+d), d = mean abs jump of z between neighbouring cells
           -> exactly zero for uniform data, so F(Q,...,Q) = f(Q)   (consistency), and the
              correction scales with the local jump like numerical dissipation does, so it cannot
              overwhelm the baseline's dissipation at weak jumps
  guard    a-posteriori local-extremum check (MOOD-style): where a trial forward-Euler update with
           the corrected flux takes rho, p or u outside [min - eps*range, max + eps*range] of the 3-cell
           neighbourhood (or non-positive), the correction is switched off on that cell's faces and
           the baseline flux is used there
  G_sym    = 0.5 * (G(z) + M G(Rz)),  R: reverse stencil & flip u,  M = diag(-1, 1, -1)
           -> exact mirror equivariance x -> -x
  last layer zero-initialised -> the untrained model *is* the baseline scheme.

Use with an SSP time integrator (SSP-RK3 / forward Euler): SSP stages are convex combinations of
forward-Euler steps, which is what the guard checks. With classical RK4 the guard's guarantee does
not carry over, and rollouts were observed to break down (see README_NEURAL.md).
'''
import torch
from torch import nn

from .torch_ops import RIEMANN_SOLVERS, reconstruct

ACTIVATIONS = {'ReLU': nn.ReLU, 'ELU': nn.ELU, 'Tanh': nn.Tanh, 'SiLU': nn.SiLU, 'GELU': nn.GELU}
DTYPES = {'float32': torch.float32, 'float64': torch.float64}


class NeuralEulerFlux(nn.Module):
    def __init__(self, stencil=4, hidden=32, depth=4, act='ELU', baseline='hllc',
                 reconstruction='first_order', symmetrize=True, max_correction=0.5,
                 n_ghost=3, gamma=1.4, dx=1.0, dt=None, positivity_guard=True, guard_slack=0.05, guard_velocity=True,
                 integrator='ssprk3', dtype='float64'):
        '''
        :param int stencil: cells seen per interface (even, <= 2*n_ghost); NFV uses 2
        :param str baseline: 'hllc' | 'rusanov' | 'central' ('central' = network supplies all dissipation)
        :param str reconstruction: reconstruction feeding the baseline flux ('first_order' | 'weno5')
        :param float dx, dt: resolution the model is trained for (stored for checks, not used as inputs)
        '''
        super().__init__()
        if stencil % 2 or not 2 <= stencil <= 2 * n_ghost:
            raise ValueError('stencil must be even and 2 <= stencil <= 2*n_ghost')
        if baseline not in RIEMANN_SOLVERS:
            raise ValueError(f'baseline must be one of {list(RIEMANN_SOLVERS)}')
        self.config = dict(stencil=stencil, hidden=hidden, depth=depth, act=act, baseline=baseline,
                           reconstruction=reconstruction, symmetrize=symmetrize,
                           max_correction=max_correction, n_ghost=n_ghost, gamma=gamma,
                           dx=dx, dt=dt, positivity_guard=positivity_guard,
                           guard_slack=guard_slack, guard_velocity=guard_velocity, integrator=integrator, dtype=dtype)
        self.stencil, self.n_ghost, self.gamma, self.dx, self.dt = stencil, n_ghost, gamma, dx, dt
        self.baseline, self.reconstruction = baseline, reconstruction
        self.symmetrize, self.max_correction = symmetrize, max_correction
        self.positivity_guard = positivity_guard and dt is not None
        self.guard_slack = guard_slack
        self.guard_velocity = guard_velocity
        self.integrator = integrator   # time integrator the flux was trained with

        tdtype, A = DTYPES[dtype], ACTIVATIONS[act]
        layers = [nn.Linear(3 * stencil, hidden, dtype=tdtype), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden, dtype=tdtype), A()]
        last = nn.Linear(hidden, 3, dtype=tdtype)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        self.register_buffer('mirror', torch.tensor([-1.0, 1.0, -1.0], dtype=tdtype))

    # ------------------------------------------------------------ pieces
    def _stencils(self, Qp):
        '''(B,3,N+2g) -> (B,3,N+1,S); interface j sits between padded cells g-1+j and g+j.'''
        g, S = self.n_ghost, self.stencil
        n_int = Qp.shape[-1] - 2 * g + 1
        start = g - S // 2
        return Qp[..., start:start + n_int + S - 1].unfold(-1, S, 1)

    def _G(self, rho, u, p, rho_ref, p_ref, c_ref):
        '''Jump-scaled correction. inputs (B,N+1,S), refs (B,N+1,1) -> (B,N+1,3).'''
        feats = torch.stack([torch.log(rho / rho_ref), u / c_ref, torch.log(p / p_ref)], dim=-2)  # (B,n,3,S)
        delta = feats.diff(dim=-1).abs().mean(dim=(-1, -2)).unsqueeze(-1)     # (B,n,1); abs, not sqrt: finite grad at 0
        delta = delta / (1.0 + delta)                                          # ~delta for weak jumps, saturates at 1
        z = feats.flatten(-2)
        return delta * self.max_correction * torch.tanh(self.net(z))

    def correction(self, Qp):
        sq = self._stencils(Qp)                                     # (B,3,n,S)
        rho = sq[:, 0].clamp_min(1e-12)
        u = sq[:, 1] / rho
        p = ((self.gamma - 1.0) * (sq[:, 2] - 0.5 * rho * u ** 2)).clamp_min(1e-12)
        c = self.stencil // 2
        rho_ref = 0.5 * (rho[..., c - 1:c] + rho[..., c:c + 1])
        p_ref = 0.5 * (p[..., c - 1:c] + p[..., c:c + 1])
        c_ref = torch.sqrt(self.gamma * p_ref / rho_ref)

        G = self._G(rho, u, p, rho_ref, p_ref, c_ref)                # (B,n,3)
        if self.symmetrize:
            Gm = self._G(rho.flip(-1), -u.flip(-1), p.flip(-1), rho_ref, p_ref, c_ref)
            G = 0.5 * (G + self.mirror * Gm)
        scale = torch.cat([rho_ref * c_ref, p_ref, p_ref * c_ref], dim=-1)
        return (scale * G).permute(0, 2, 1)                          # (B,3,n)

    def base_flux(self, Qp):
        QL, QR = reconstruct(Qp, self.reconstruction, self.n_ghost, self.dx)
        return RIEMANN_SOLVERS[self.baseline](QL, QR, self.gamma)

    # ------------------------------------------------------------ forward
    def forward(self, Qp):
        '''Qp: (B,3,N+2g) ghost-padded conservative state -> F: (B,3,N+1)'''
        Fb, C = self.base_flux(Qp), self.correction(Qp)
        if self.positivity_guard:
            C = C * self._guard_mask(Qp, Fb, C)
        return Fb + C

    @torch.no_grad()
    def _guard_mask(self, Qp, Fb, C, max_iter=50):
        '''
        Face mask (B,1,N+1): 1 keeps the learned correction, 0 reverts the face to the baseline flux.

        A cell passes if a trial forward-Euler step (training dt, dx) keeps rho and p positive and
        rho, p (and u, if guard_velocity) inside [min - s*range, max + s*range] of the old 3-cell
        neighbourhood (s = guard_slack). Checking u matters at sonic points (transonic rarefactions),
        where removing dissipation produces velocity overshoots without new rho/p extrema per step:
        the correction may sharpen profiles but not create new extrema. A bound relative to the
        current state only (e.g. "p_new > 0.5 p_min") is NOT enough: it compounds over steps and
        lets a contact drift toward vacuum.
        Reverting a failing cell's faces changes its neighbours' updates, so the check is repeated
        until no further face is reverted (the mask only shrinks; its limit is the baseline scheme).
        '''
        g, gm, s = self.n_ghost, self.gamma, self.guard_slack
        Q = Qp[..., g:-g]
        lam = self.dt / self.dx

        def rho_p(q):
            return q[:, 0], (gm - 1.0) * (q[:, 2] - 0.5 * q[:, 1] ** 2 / q[:, 0])

        def vel(q):
            return q[:, 1] / q[:, 0]

        def bounds(a):                                                   # (B,N+2) -> lo, hi (B,N)
            nb = torch.stack([a[:, :-2], a[:, 1:-1], a[:, 2:]])
            lo, hi = nb.min(0).values, nb.max(0).values
            return lo - s * (hi - lo), hi + s * (hi - lo)

        r_old, p_old = rho_p(Qp[..., g - 1:Qp.shape[-1] - g + 1])      # cells incl. one ghost each side
        (r_lo, r_hi), (p_lo, p_hi) = bounds(r_old), bounds(p_old)
        if self.guard_velocity:
            u_lo, u_hi = bounds(vel(Qp[..., g - 1:Qp.shape[-1] - g + 1]))

        keep = torch.ones_like(Fb[:, :1])                                # (B,1,N+1)
        for _ in range(max_iter):
            F = Fb + keep * C
            Qn = Q - lam * (F[..., 1:] - F[..., :-1])
            r_new, p_new = rho_p(Qn)
            ok = (torch.isfinite(r_new) & torch.isfinite(p_new) & (r_new > 0) & (p_new > 0)
                  & (r_new >= r_lo) & (r_new <= r_hi) & (p_new >= p_lo) & (p_new <= p_hi))
            if self.guard_velocity:
                u_new = vel(Qn)
                ok = ok & (u_new >= u_lo) & (u_new <= u_hi)
            bad = torch.nn.functional.pad((~ok).to(Q.dtype), (1, 1))    # (B,N+2)
            new_keep = keep * (1.0 - torch.maximum(bad[:, :-1], bad[:, 1:])).unsqueeze(1)
            if torch.equal(new_keep, keep):
                break
            keep = new_keep
        return keep

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_neural_flux(model, path, extra=None):
    torch.save({'config': model.config, 'state_dict': model.state_dict(), 'extra': extra or {}}, path)


def load_neural_flux(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = NeuralEulerFlux(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()
