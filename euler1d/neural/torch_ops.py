'''
Differentiable (PyTorch) counterparts of the numpy building blocks in euler1d/.

Everything here operates on a *batched* conservative state
    Q : (B, 3, N)      rows = [rho, rho*u, E]
so that the finite-volume update can be unrolled and back-propagated through
(the core training idea of NFV). Each function is numerically checked against the
numpy implementation in tests/test_neural_integration.py.
'''
import torch

EPS = 1e-12


def cons_to_prim(Q, gamma=1.4):
    rho = Q[:, 0]
    u = Q[:, 1] / rho
    p = (gamma - 1.0) * (Q[:, 2] - 0.5 * rho * u ** 2)
    return rho, u, p


def prim_to_cons(rho, u, p, gamma=1.4):
    return torch.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u ** 2], dim=1)


def sound_speed(rho, p, gamma=1.4):
    return torch.sqrt(gamma * p.clamp_min(EPS) / rho.clamp_min(EPS))


def physical_flux(Q, gamma=1.4):
    rho, u, p = cons_to_prim(Q, gamma)
    return torch.stack([rho * u, rho * u ** 2 + p, u * (Q[:, 2] + p)], dim=1)


def central_flux(QL, QR, gamma=1.4):
    '''Arithmetic mean of physical fluxes (no dissipation; unstable on its own).'''
    return 0.5 * (physical_flux(QL, gamma) + physical_flux(QR, gamma))


def rusanov_flux(QL, QR, gamma=1.4):
    '''Local Lax-Friedrichs flux.'''
    rL, uL, pL = cons_to_prim(QL, gamma)
    rR, uR, pR = cons_to_prim(QR, gamma)
    s = torch.maximum(uL.abs() + sound_speed(rL, pL, gamma), uR.abs() + sound_speed(rR, pR, gamma))
    return central_flux(QL, QR, gamma) - 0.5 * s.unsqueeze(1) * (QR - QL)


def hllc_flux(QL, QR, gamma=1.4):
    '''Torch port of euler1d/schemes/flux/hllc.py (same wave-speed estimates and star states).'''
    rL, uL, pL = cons_to_prim(QL, gamma)
    rR, uR, pR = cons_to_prim(QR, gamma)
    aL, aR = sound_speed(rL, pL, gamma), sound_speed(rR, pR, gamma)
    FL, FR = physical_flux(QL, gamma), physical_flux(QR, gamma)

    SL = torch.minimum(uL - aL, uR - aR)
    SR = torch.maximum(uL + aL, uR + aR)
    # denominators below are nonzero for physical states (SL-uL <= -aL < 0 < aR <= SR-uR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def star(Q, r, u, p, S):
        coef = r * (S - u) / (S - Ss)
        row2 = Q[:, 2] / r + (Ss - u) * (Ss + p / (r * (S - u)))
        return coef.unsqueeze(1) * torch.stack([torch.ones_like(Ss), Ss, row2], dim=1)

    FLs = FL + SL.unsqueeze(1) * (star(QL, rL, uL, pL, SL) - QL)
    FRs = FR + SR.unsqueeze(1) * (star(QR, rR, uR, pR, SR) - QR)
    SL_, SR_, Ss_ = SL.unsqueeze(1), SR.unsqueeze(1), Ss.unsqueeze(1)
    return torch.where(SL_ >= 0, FL, torch.where(SR_ <= 0, FR, torch.where(Ss_ >= 0, FLs, FRs)))


RIEMANN_SOLVERS = {'central': central_flux, 'rusanov': rusanov_flux, 'hllc': hllc_flux}


# ghost cells
def apply_bc(Q, bc='Zero-Gradient', n_ghost=3, ghosts=None):
    '''
    Torch port of euler1d/boundary.py.  Q: (B,3,N) -> (B,3,N+2*n_ghost)

    If `ghosts` (B,3,2*n_ghost) is given, those values are used instead of `bc`
    (NFV-style pinning of the boundary to ground truth).
    '''
    g = n_ghost
    if ghosts is not None:
        left, right = ghosts[..., :g], ghosts[..., g:]
    elif bc == 'Zero-Gradient':
        left = Q[..., :1].expand(-1, -1, g)
        right = Q[..., -1:].expand(-1, -1, g)
    elif bc == 'Reflective':
        sign = torch.tensor([1.0, -1.0, 1.0], dtype=Q.dtype, device=Q.device).view(1, 3, 1)
        left = Q[..., :g].flip(-1) * sign
        right = Q[..., -g:].flip(-1) * sign
    elif bc == 'Periodical':
        left, right = Q[..., -g:], Q[..., :g]
    else:
        raise ValueError(f'Invalid BC {bc}')
    return torch.cat([left, Q, right], dim=-1)


# ---------------------------------------------------------------- reconstruction
def weno5_js(Qp, dx=1.0):
    '''Torch port of euler1d/schemes/reconstruction/weno5.py (component-wise, JS weights).'''
    n = Qp.shape[-1] - 6

    def s(k):
        return Qp[..., k:k + n + 1]

    def smooth(im2, im1, i, ip1, ip2):
        b0 = 13 / 12 * (i - 2 * ip1 + ip2) ** 2 + 1 / 4 * (3 * i - 4 * ip1 + ip2) ** 2
        b1 = 13 / 12 * (im1 - 2 * i + ip1) ** 2 + 1 / 4 * (im1 - ip1) ** 2
        b2 = 13 / 12 * (im2 - 2 * im1 + i) ** 2 + 1 / 4 * (im2 - 4 * im1 + 3 * i) ** 2
        return b0, b1, b2

    def weights(d, betas):
        eps = 1e-6 * dx ** 2
        alphas = [dk / (eps + bk) ** 2 for dk, bk in zip(d, betas)]
        tot = alphas[0] + alphas[1] + alphas[2]
        return [a / tot for a in alphas]

    im2, im1, i, ip1, ip2 = (s(k) for k in range(5))
    wL = weights((3 / 10, 3 / 5, 1 / 10), smooth(im2, im1, i, ip1, ip2))
    QL = (wL[0] * (1 / 3 * i + 5 / 6 * ip1 - 1 / 6 * ip2)
          + wL[1] * (-1 / 6 * im1 + 5 / 6 * i + 1 / 3 * ip1)
          + wL[2] * (1 / 3 * im2 - 7 / 6 * im1 + 11 / 6 * i))

    im2, im1, i, ip1, ip2 = (s(k) for k in range(1, 6))
    wR = weights((1 / 10, 3 / 5, 3 / 10), smooth(im2, im1, i, ip1, ip2))
    QR = (wR[0] * (11 / 6 * i - 7 / 6 * ip1 + 1 / 3 * ip2)
          + wR[1] * (1 / 3 * im1 + 5 / 6 * i - 1 / 6 * ip1)
          + wR[2] * (-1 / 6 * im2 + 5 / 6 * im1 + 1 / 3 * i))

    QL = torch.cat([QL[:, :1].clamp_min(1e-10), QL[:, 1:]], dim=1)
    QR = torch.cat([QR[:, :1].clamp_min(1e-10), QR[:, 1:]], dim=1)
    return QL, QR


def reconstruct(Qp, method='first_order', n_ghost=3, dx=1.0):
    '''Qp: (B,3,N+2g) -> QL, QR: (B,3,N+1)'''
    g = n_ghost
    n = Qp.shape[-1] - 2 * g
    if method == 'first_order':
        return Qp[..., g - 1:g + n], Qp[..., g:g + n + 1]
    if method == 'weno5':
        if g != 3:
            raise ValueError('weno5 requires n_ghost == 3')
        return weno5_js(Qp, dx)
    raise ValueError(f'Unknown reconstruction {method}')
