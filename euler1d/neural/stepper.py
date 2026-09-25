'''
Differentiable finite-volume time stepping for the 1D Euler equations.

A "flux operator" is any callable  Qp (B,3,N+2g) -> F (B,3,N+1):
  * ClassicalFlux  (torch reconstruction + Riemann solver), used as baselines
  * NeuralEulerFlux (euler1d/neural/model.py)
'''
import torch
from torch import nn

from .torch_ops import RIEMANN_SOLVERS, apply_bc, reconstruct


class ClassicalFlux(nn.Module):
    def __init__(self, reconstruction='first_order', riemann='hllc', gamma=1.4, n_ghost=3, dx=1.0):
        super().__init__()
        self.reconstruction, self.riemann = reconstruction, riemann
        self.gamma, self.n_ghost, self.dx = gamma, n_ghost, dx

    def forward(self, Qp):
        QL, QR = reconstruct(Qp, self.reconstruction, self.n_ghost, self.dx)
        return RIEMANN_SOLVERS[self.riemann](QL, QR, self.gamma)

    def __repr__(self):
        return f'ClassicalFlux({self.reconstruction}, {self.riemann})'


def fv_rhs(Q, flux_op, dx, bc='Zero-Gradient', n_ghost=3, ghosts=None):
    '''Semi-discrete RHS  dQ/dt = -(F_{i+1/2} - F_{i-1/2}) / dx'''
    F = flux_op(apply_bc(Q, bc, n_ghost, ghosts))
    return -(F[..., 1:] - F[..., :-1]) / dx


def step(Q, dt, dx, flux_op, bc='Zero-Gradient', n_ghost=3, ghosts=None, integrator='rk4'):
    '''One time step. Ghost values (if given) are frozen over the RK stages.'''
    def f(q):
        return fv_rhs(q, flux_op, dx, bc, n_ghost, ghosts)

    if integrator == 'euler':  # what NFV uses
        return Q + dt * f(Q)
    if integrator == 'ssprk3':
        q1 = Q + dt * f(Q)
        q2 = 0.75 * Q + 0.25 * (q1 + dt * f(q1))
        return Q / 3.0 + 2.0 / 3.0 * (q2 + dt * f(q2))
    if integrator == 'rk4':  # what euler1d/solver.py uses
        k1 = dt * f(Q)
        k2 = dt * f(Q + k1 / 2)
        k3 = dt * f(Q + k2 / 2)
        k4 = dt * f(Q + k3)
        return Q + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    raise ValueError(f'Unknown integrator {integrator}')


def rollout(Q0, n_steps, dt, dx, flux_op, bc='Zero-Gradient', n_ghost=3, ghosts_BT=None, integrator='rk4'):
    '''
    Autoregressive rollout. Q0: (B,3,N); ghosts_BT: optional (B,T,3,2g) with T >= n_steps.
    Returns (B, n_steps+1, 3, N).
    '''
    Q, hist = Q0, [Q0]
    for t in range(n_steps):
        g_t = ghosts_BT[:, t] if ghosts_BT is not None else None
        Q = step(Q, dt, dx, flux_op, bc, n_ghost, g_t, integrator)
        hist.append(Q)
    return torch.stack(hist, dim=1)
