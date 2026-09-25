'''
Bridge between the numpy EulerSolver and a torch NeuralEulerFlux.

EulerSolver calls   F = scheme(Q_num, dx=dx, gamma=gamma)
with Q_num the ghost-padded state (3, N + 2*N_ghost), and expects F with shape (3, N+1).
'''
import warnings

import numpy as np
import torch

from .model import NeuralEulerFlux, load_neural_flux


class NeuralFluxScheme:
    def __init__(self, model_or_path, device='cpu'):
        if isinstance(model_or_path, NeuralEulerFlux):
            self.model = model_or_path.to(device).eval()
        else:
            self.model = load_neural_flux(model_or_path, device)
        self.device = device
        self.dtype = next(self.model.parameters()).dtype
        self.required_ghost = self.model.n_ghost
        self._warned = False

    def check_resolution(self, dx, dt=None):
        '''A learned flux is calibrated to the (dx, dt) it was trained at (as in NFV).'''
        msgs = []
        if self.model.dx and abs(dx / self.model.dx - 1.0) > 1e-2:
            msgs.append(f'dx={dx:g} (trained at {self.model.dx:g})')
        if dt is not None and self.model.dt and abs(dt / self.model.dt - 1.0) > 1e-2:
            msgs.append(f'dt={dt:g} (trained at {self.model.dt:g})')
        if msgs:
            warnings.warn('Neural flux used off its training resolution: ' + ', '.join(msgs))

    def __call__(self, Q_num, dx=None, gamma=None):
        if gamma is not None and abs(gamma - self.model.gamma) > 1e-12:
            raise ValueError(f'gamma={gamma} but the neural flux was trained with gamma={self.model.gamma}')
        if not self._warned and dx is not None:
            self.check_resolution(dx)
            self._warned = True
        with torch.no_grad():
            Qt = torch.as_tensor(np.asarray(Q_num), dtype=self.dtype, device=self.device).unsqueeze(0)
            return self.model(Qt)[0].cpu().numpy().astype(np.float64)
