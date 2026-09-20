import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import root_scalar

from euler1d.prob_config import EulerConfig
from euler1d.solver import EulerSolver
from euler1d.results import animate_results


'''
I'm sorry i just wanted to test the damn thing cuz its nearly bigwalk time and i wanna get wine drunk


i promise every other file is handcoded
'''

# =============================================================================
# Exact Riemann Solver for 1D Euler Equations
# =============================================================================
class ExactRiemannSolver:
    """
    Computes the exact solution to the 1D Euler Riemann problem at any (x, t).
    """
    def __init__(self, state_L, state_R, x_interface=0.5, gamma=1.4):
        self.rho_L, self.u_L, self.P_L = state_L
        self.rho_R, self.u_R, self.P_R = state_R
        self.x_int = x_interface
        self.gamma = gamma

        self.a_L = np.sqrt(gamma * self.P_L / self.rho_L)
        self.a_R = np.sqrt(gamma * self.P_R / self.rho_R)

        # Solve for star region pressure (P_star) and velocity (u_star)
        self.P_star = self._solve_p_star()
        self.u_star = 0.5 * (self.u_L + self.u_R) + 0.5 * (self._f(self.P_star, 'R') - self._f(self.P_star, 'L'))

        # Star densities
        self.rho_star_L = self._calc_rho_star(self.P_star, 'L')
        self.rho_star_R = self._calc_rho_star(self.P_star, 'R')

    def _f_k(self, P, state_type):
        P_k = self.P_L if state_type == 'L' else self.P_R
        rho_k = self.rho_L if state_type == 'L' else self.rho_R
        a_k = self.a_L if state_type == 'L' else self.a_R
        g = self.gamma

        if P > P_k:  # Shock wave
            A_k = 2.0 / ((g + 1.0) * rho_k)
            B_k = (g - 1.0) / (g + 1.0) * P_k
            return (P - P_k) * np.sqrt(A_k / (P + B_k))
        else:        # Rarefaction wave
            return (2.0 * a_k / (g - 1.0)) * ((P / P_k)**((g - 1.0) / (2.0 * g)) - 1.0)

    def _f(self, P, state_type):
        return self._f_k(P, state_type)

    def _solve_p_star(self):
        # Two-shock approximation initial guess for root finder
        g = self.gamma
        p_pv = max(1e-6, 0.5 * (self.P_L + self.P_R) - 0.125 * (self.u_R - self.u_L) * (self.rho_L + self.rho_R) * (self.a_L + self.a_R))
        
        func = lambda P: self._f_k(P, 'L') + self._f_k(P, 'R') + (self.u_R - self.u_L)
        
        try:
            res = root_scalar(func, x0=p_pv, bracket=[1e-8, max(self.P_L, self.P_R) * 10.0], method='brentq')
            return res.root
        except ValueError:
            return max(1e-6, p_pv)

    def _calc_rho_star(self, P_star, state_type):
        P_k = self.P_L if state_type == 'L' else self.P_R
        rho_k = self.rho_L if state_type == 'L' else self.rho_R
        g = self.gamma

        if P_star > P_k:  # Shock
            return rho_k * ((P_star / P_k + (g - 1.0) / (g + 1.0)) / 
                           ((g - 1.0) / (g + 1.0) * (P_star / P_k) + 1.0))
        else:             # Rarefaction
            return rho_k * (P_star / P_k)**(1.0 / g)

    def sample(self, x_grid, t):
        if t <= 1e-12:
            rho = np.where(x_grid < self.x_int, self.rho_L, self.rho_R)
            u   = np.where(x_grid < self.x_int, self.u_L, self.u_R)
            P   = np.where(x_grid < self.x_int, self.P_L, self.P_R)
            return rho, u, P

        S = (x_grid - self.x_int) / t
        g = self.gamma
        
        rho = np.zeros_like(x_grid)
        u   = np.zeros_like(x_grid)
        P   = np.zeros_like(x_grid)

        for i, s in enumerate(S):
            if s < self.u_star:  # Left of contact discontinuity
                if self.P_star > self.P_L:  # Left Shock
                    S_L = self.u_L - self.a_L * np.sqrt((g + 1.0) / (2.0 * g) * (self.P_star / self.P_L) + (g - 1.0) / (2.0 * g))
                    if s < S_L:
                        rho[i], u[i], P[i] = self.rho_L, self.u_L, self.P_L
                    else:
                        rho[i], u[i], P[i] = self.rho_star_L, self.u_star, self.P_star
                else:  # Left Rarefaction
                    SH_L = self.u_L - self.a_L
                    a_star_L = self.a_L * (self.P_star / self.P_L)**((g - 1.0) / (2.0 * g))
                    ST_L = self.u_star - a_star_L
                    if s < SH_L:
                        rho[i], u[i], P[i] = self.rho_L, self.u_L, self.P_L
                    elif s > ST_L:
                        rho[i], u[i], P[i] = self.rho_star_L, self.u_star, self.P_star
                    else:  # Inside fan
                        u[i] = 2.0 / (g + 1.0) * (self.a_L + (g - 1.0) / 2.0 * self.u_L + s)
                        a_fan = 2.0 / (g + 1.0) * (self.a_L + (g - 1.0) / 2.0 * (self.u_L - s))
                        rho[i] = self.rho_L * (a_fan / self.a_L)**(2.0 / (g - 1.0))
                        P[i] = self.P_L * (a_fan / self.a_L)**(2.0 * g / (g - 1.0))

            else:  # Right of contact discontinuity
                if self.P_star > self.P_R:  # Right Shock
                    S_R = self.u_R + self.a_R * np.sqrt((g + 1.0) / (2.0 * g) * (self.P_star / self.P_R) + (g - 1.0) / (2.0 * g))
                    if s > S_R:
                        rho[i], u[i], P[i] = self.rho_R, self.u_R, self.P_R
                    else:
                        rho[i], u[i], P[i] = self.rho_star_R, self.u_star, self.P_star
                else:  # Right Rarefaction
                    SH_R = self.u_R + self.a_R
                    a_star_R = self.a_R * (self.P_star / self.P_R)**((g - 1.0) / (2.0 * g))
                    ST_R = self.u_star + a_star_R
                    if s > SH_R:
                        rho[i], u[i], P[i] = self.rho_R, self.u_R, self.P_R
                    elif s < ST_R:
                        rho[i], u[i], P[i] = self.rho_star_R, self.u_star, self.P_star
                    else:  # Inside fan
                        u[i] = 2.0 / (g + 1.0) * (-self.a_R + (g - 1.0) / 2.0 * self.u_R + s)
                        a_fan = 2.0 / (g + 1.0) * (self.a_R - (g - 1.0) / 2.0 * (self.u_R - s))
                        rho[i] = self.rho_R * (a_fan / self.a_R)**(2.0 / (g - 1.0))
                        P[i] = self.P_R * (a_fan / self.a_R)**(2.0 * g / (g - 1.0))

        return rho, u, P


# =============================================================================
# Custom Animation function comparing Numerical vs Exact Solution
# =============================================================================
def animate_comparison(results, exact_solver, interval=30):
    N_cells = results.config.N_cells
    domain_size = results.config.domain_size
    dx = results.config.dx
    gamma = results.config.gamma

    x_num = np.linspace(dx / 2.0, domain_size - dx / 2.0, N_cells)
    x_exact = np.linspace(0, domain_size, 1000)

    # Compute numerical state history
    rho_num_hist, u_num_hist, p_num_hist = [], [], []
    for Q in results.Q_hist:
        rho = Q[0, :]
        u = Q[1, :] / rho
        P = (gamma - 1.0) * (Q[2, :] - 0.5 * rho * u**2)

        rho_num_hist.append(rho)
        u_num_hist.append(u)
        p_num_hist.append(P)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle('1D Euler Solver: WENO5Z/HLLC vs Exact Analytical', fontsize=13, fontweight='bold')

    # Numerical lines (scatter/lines)
    line_rho_num, = ax1.plot([], [], 'b-o', ms=3, lw=1.5, label='Numerical (WENO5Z)')
    line_u_num,   = ax2.plot([], [], 'r-o', ms=3, lw=1.5, label='Numerical (WENO5Z)')
    line_p_num,   = ax3.plot([], [], 'g-o', ms=3, lw=1.5, label='Numerical (WENO5Z)')

    # Exact solution lines (dashed)
    line_rho_exact, = ax1.plot([], [], 'k--', lw=2, label='Exact Analytical')
    line_u_exact,   = ax2.plot([], [], 'k--', lw=2, label='Exact Analytical')
    line_p_exact,   = ax3.plot([], [], 'k--', lw=2, label='Exact Analytical')

    for ax, title in zip([ax1, ax2, ax3], ['Density (kg/m³)', 'Velocity (m/s)', 'Pressure (Pa)']):
        ax.set_ylabel(title)
        ax.set_xlim(0, domain_size)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    time_text = ax1.text(0.02, 0.82, '', transform=ax1.transAxes, 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Auto scale y limits based on max values
    rho_all = np.concatenate(rho_num_hist)
    u_all   = np.concatenate(u_num_hist)
    p_all   = np.concatenate(p_num_hist)

    ax1.set_ylim(min(0, np.min(rho_all)), np.max(rho_all) * 1.15)
    ax2.set_ylim(min(-10, np.min(u_all) * 1.1), max(10, np.max(u_all) * 1.1))
    ax3.set_ylim(min(0, np.min(p_all)), np.max(p_all) * 1.15)
    ax3.set_xlabel('x (m)')

    def init():
        for l in [line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact]:
            l.set_data([], [])
        time_text.set_text('')
        return line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact, time_text

    def update(frame):
        t = results.t_hist[frame]

        # Numerical update
        line_rho_num.set_data(x_num, rho_num_hist[frame])
        line_u_num.set_data(x_num, u_num_hist[frame])
        line_p_num.set_data(x_num, p_num_hist[frame])

        # Exact update
        rho_ex, u_ex, p_ex = exact_solver.sample(x_exact, t)
        line_rho_exact.set_data(x_exact, rho_ex)
        line_u_exact.set_data(x_exact, u_ex)
        line_p_exact.set_data(x_exact, p_ex)

        time_text.set_text(f't = {t:.5f} s')
        return line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact, time_text

    anim = FuncAnimation(
        fig, update, frames=len(results.t_hist),
        init_func=init, blit=True, interval=interval
    )

    plt.tight_layout()
    plt.show()
    return anim


# =============================================================================
# Test Runner
# =============================================================================
def test_riemann_problem():
    """
    Runs Sod Shock Tube problem test case to compare against exact solution.
    """
    N_cells = 200
    domain_size = 1.0
    t_max = 0.0007
    dt = 1e-6
    N_ghost = 3
    gamma = 1.4

    flux_scheme = 'HLLC'
    reconstruction_scheme = 'WENO5Z'

    # Standard Sod Shock Tube problem (or custom initial states)
    rho_L, u_L, P_L = 1.0, 0.0, 100000.0
    rho_R, u_R, P_R = 0.125, 0.0, 10000.0

    # Primitive states for Exact Solver
    state_L = (rho_L, u_L, P_L)
    state_R = (rho_R, u_R, P_R)

    def get_cons(rho, u, P, gamma_val):
        mom = rho * u
        E = (P / (gamma_val - 1.0)) + 0.5 * rho * u**2
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
    test_riemann_problem()