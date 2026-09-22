import numpy as np
from infrastructure.solver_config import EulerConfig
from euler1d.solver import EulerSolver
from infrastructure.results import ExactRiemannSolver, animate_comparison


'''
Gemini-Produced Sod Shock Tube Problem
    - Modifications made by pranet
'''


def test_Sod_Shock():
    """
    Runs Sod Shock Tube problem test case to compare against exact solution.
    """
    N_cells = 1000
    domain_size = 1.0
    t_max = 0.001
    dt = 1e-6
    N_ghost = 3
    gamma = 1.4

    '''
    SCHEME SELECTION
    '''

    flux_scheme = 'HLLC'
    reconstruction_scheme = 'WENO5Z'

    # flux_scheme = 'AUSM+'
    # reconstruction_scheme = 'WENO5'

    # Standard Sod Shock Tube problem (or custom initial states)
    rho_L, u_L, P_L = 1.0, 0, 100000.0
    rho_R, u_R, P_R = 0.125, 0, 10000.0

    # Primitive states for Exact Solver
    state_L = (rho_L, u_L, P_L)
    state_R = (rho_R, u_R, P_R)

    def get_cons(rho, u, P, gamma):
        mom = rho * u
        E = (P / (gamma - 1.0)) + 0.5 * rho * u**2
        return np.array([rho, mom, E])

    Q_L = get_cons(rho_L, u_L, P_L, gamma)
    Q_R = get_cons(rho_R, u_R, P_R, gamma)

    # Initial Condition Matrix (3, N_cells)
    IC = np.zeros((3, N_cells))
    x_split = N_cells // 2
    for i in range(N_cells):
        IC[:, i] = Q_L if i < x_split else Q_R

    config = EulerConfig(
        domain_size=domain_size,
        N_cells=N_cells,
        IC=IC,
        BC='Zero-Gradient',
        t_max=t_max,
        dt=dt,
        gamma=gamma,
        N_ghost=N_ghost
    )

    print(f"Running EulerSolver with {flux_scheme} & {reconstruction_scheme}...")
    results = EulerSolver(
        config=config,
        flux=flux_scheme,
        reconstruction=reconstruction_scheme
    )

    # Instantiate Exact Solver
    exact_solver = ExactRiemannSolver(state_L, state_R, x_interface=domain_size / 2.0, gamma=gamma)

    # Animate
    animate_comparison(results, exact_solver)


if __name__ == '__main__':
    test_Sod_Shock()