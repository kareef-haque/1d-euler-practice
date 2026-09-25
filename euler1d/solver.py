'''
1D Euler Equation Solver

- Two Flux Reconstruction Methods
    - Kareef: AUSM + & WENO 5
    - Pranet: HLLC & WENO 5Z
- Neural Finite Volume flux (NFV-style, trained with euler1d/neural/train.py)
    - flux = 'Neural', neural_scheme = path to a .pt checkpoint (or a NeuralFluxScheme)
    - the neural scheme sees the ghost-padded state directly, so `reconstruction` is ignored

Specify simulation parameters via ...
Specify solver parameters via...
'''

import os
import numpy as np
from euler1d.boundary import apply_BC
from infrastructure.solver_config import EulerConfig
from infrastructure.results import EulerResults
from datetime import datetime

from euler1d.schemes.flux.hllc import HLLC
from euler1d.schemes.flux.ausm_plus import AUSMp

from euler1d.schemes.reconstruction.weno5Z import WENO5Z
from euler1d.schemes.reconstruction.weno5 import WENO5
from euler1d.schemes.reconstruction.first_order import FirstOrder

NEURAL_FLUX_NAMES = ('Neural', 'NFV')


def _load_neural_scheme(neural_scheme):
    # imported lazily so classical runs do not require PyTorch
    from euler1d.neural.adapter import NeuralFluxScheme
    if neural_scheme is None:
        raise ValueError("flux='Neural' requires neural_scheme (checkpoint path or NeuralFluxScheme)")
    if isinstance(neural_scheme, (str, os.PathLike)):
        return NeuralFluxScheme(neural_scheme)
    return neural_scheme


def EulerSolver(config: EulerConfig,
                flux = 'HLLC',
                reconstruction = 'WENO5Z',
                neural_scheme = None,
                time_integrator = None,
                verbose = True
                ):
    '''
    :param EulerConfig config: Configuration for the problem
    :param str flux: Flux Scheme (AUSM+, HLLC, Neural)
    :param str reconstruction: Reconstruction Method (WENO5Z, WENO5, FirstOrder); ignored for Neural
    :param neural_scheme: checkpoint path or NeuralFluxScheme, required when flux == 'Neural'
    :param str time_integrator: 'RK4' or 'SSPRK3'. Default: RK4 for classical schemes, and the
        integrator the model was trained with for 'Neural' (SSPRK3 unless trained otherwise)

    :return EulerResults: Results of the simulation

    Main function to run the 1D Euler simulation
        - integrates RK4 for time stepping
        - uses specified flux and reconstruction schemes
    '''
    #unpack config 
    domain_size = config.domain_size
    N_cells = config.N_cells
    dx = config.dx
    t_max = config.t_max
    dt = config.dt
    gamma = config.gamma

    Q_init = config.IC
    BC = config.BC
    N_ghost = config.N_ghost

    #Select schemes based on user input
    if flux in NEURAL_FLUX_NAMES:
        scheme = _load_neural_scheme(neural_scheme)
        if N_ghost < scheme.required_ghost:
            raise ValueError(f'Neural scheme needs N_ghost >= {scheme.required_ghost}')
        scheme.check_resolution(dx, dt)
        if time_integrator is None:
            time_integrator = {'ssprk3': 'SSPRK3', 'rk4': 'RK4'}.get(getattr(scheme.model, 'integrator', 'rk4'), 'RK4')

        def calc_Flux(Q_curr):
            Q_num = apply_BC(Q = Q_curr, BC = BC, ghost_N = N_ghost)
            return scheme(Q_num, dx = dx, gamma = gamma)
    else:
        if flux == 'HLLC':
            F_scheme = HLLC
        elif flux == 'AUSM+' or flux == 'AUSMp':
            F_scheme = AUSMp
        else:
            raise ValueError("Invalid Flux Scheme")

        if reconstruction == 'WENO5Z':
            R_scheme = WENO5Z
        elif reconstruction == 'WENO5':
            R_scheme = WENO5
        elif reconstruction == 'FirstOrder':
            R_scheme = FirstOrder
        else:
            raise ValueError("Invalid Reconstruction Scheme")

        def calc_Flux(Q_curr):
            Q_num = apply_BC(Q = Q_curr,
                             BC = BC,
                             ghost_N = N_ghost)

            Q_L, Q_R = R_scheme(Q_num, dx = dx)
            F = F_scheme(Q_L, Q_R, dx = dx, gamma = gamma)
            return F

    time_integrator = time_integrator or 'RK4'
    if time_integrator not in ('RK4', 'SSPRK3'):
        raise ValueError("time_integrator must be 'RK4' or 'SSPRK3'")

    # RHS of discretized 1D Euler
    def calc_discrete_Q_dot(Q_set, dx = dx):
        F = calc_Flux(Q_set)
        return -1/dx * (F[:, 1:] - F[:, :-1])
    
    t_hist = [0.0]
    Q_hist = [Q_init]
    F_hist = [calc_Flux(Q_init)]

    t_curr = 0.0
    iteration_count = 0
    start_time = datetime.now()
    while t_curr < t_max * (1 - 1e-12):
        dt_step = min(dt, t_max - t_curr)

        Q_curr = Q_hist[-1]

        if time_integrator == 'RK4':
            #RK 4 Implementation 
                # Note that discrete Q_dot is independent of time
            k1 = dt_step*calc_discrete_Q_dot(Q_curr)
            k2 = dt_step*calc_discrete_Q_dot(Q_curr+k1/2)
            k3 = dt_step*calc_discrete_Q_dot(Q_curr+k2/2)
            k4 = dt_step*calc_discrete_Q_dot(Q_curr+k3)

            # Update next value of y
            Q_next = Q_curr + (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)
        else:
            # SSP-RK3 (Shu-Osher): convex combination of forward-Euler steps
            Q1 = Q_curr + dt_step*calc_discrete_Q_dot(Q_curr)
            Q2 = 0.75*Q_curr + 0.25*(Q1 + dt_step*calc_discrete_Q_dot(Q1))
            Q_next = Q_curr/3.0 + (2.0/3.0)*(Q2 + dt_step*calc_discrete_Q_dot(Q2))

        Q_hist.append(Q_next)
        t_curr += dt_step
        t_hist.append(t_curr)
        F = calc_Flux(Q_next)
        F_hist.append(F)

        # Progress Monitoring: Increment counter and log every N iterations
        iteration_count += 1
        if verbose and iteration_count % 10 == 0:
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            print(f"Progress: {iteration_count} | iterations ({elapsed_seconds:.3f}s) | Sim Time ({t_curr:.6f}s): ")

    if verbose:
        print(f"Simulation Complete")
    Results = EulerResults(Q_hist,
                           F_hist,
                           t_hist,
                           config)

    return Results
