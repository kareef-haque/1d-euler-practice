'''
Run randomly generated Shock Tube Problem solved via Euler Solver
- Essentially a copy and paste of the test_Sod_Shock function, but why not
- Also useful for the data_generation code
- Can also run a trained neural flux (flux_scheme='Neural', neural_model='runs/.../model_best.pt');
  note a neural flux is only calibrated for the dx/dt it was trained at
'''

import numpy as np
from infrastructure.solver_config import EulerConfig
from infrastructure.physics_config import PhysicsConfig
from euler1d.solver import EulerSolver
from infrastructure.results import ExactRiemannSolver, animate_comparison




def random_Run(flux_scheme = 'HLLC',
               reconstruction_scheme = 'WENO5Z',
               neural_model = None,
               N_cells = 1000,
               dt = 1e-6,
               return_physics = False):
    """
    Runs a randomly generated shock tube problem.

    :param str flux_scheme: 'HLLC', 'AUSM+' or 'Neural'
    :param str reconstruction_scheme: 'WENO5Z', 'WENO5' or 'FirstOrder' (ignored for 'Neural')
    :param neural_model: checkpoint path / NeuralFluxScheme, required for 'Neural'
    :param bool return_physics: also return the PhysicsConfig (needed to save metadata)
    """

    #NOTE: THESE KEY SOLVER PARAMETERS ARE FIXED ACROSS ALL SIMS
    domain_size = 1.0
    t_max = 0.00075
    N_ghost = 3
    gamma = 1.4

    #NOTE: SCHEME SELECTION HERE (NO-DUH)
    # flux_scheme = 'AUSM+'
    # reconstruction_scheme = 'WENO5'

    physics_config = PhysicsConfig(
        BC='Zero-Gradient',
        N_cells=N_cells,
        gamma=gamma
    )

    solver_config = EulerConfig(
        domain_size=domain_size,
        N_cells=N_cells,
        IC=physics_config.IC,
        BC=physics_config.BC,
        t_max=t_max,
        dt=dt,
        gamma=gamma,
        N_ghost=N_ghost
    )

    print(f"Running EulerSolver with {flux_scheme} & {reconstruction_scheme}...")
    results = EulerSolver(
        config=solver_config,
        flux=flux_scheme,
        reconstruction=reconstruction_scheme,
        neural_scheme=neural_model
    )

    if return_physics:
        return results, physics_config
    return results



if __name__ == '__main__':
    random_Run()
