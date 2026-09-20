'''
1D Euler Solver Results Hub

Stores Results from the 1D Euler Solver into EulerResults Class
Also contains visualizaton functions
'''

from euler1d.prob_config import EulerConfig

from dataclasses import dataclass, field
import numpy as np

@dataclass
class EulerResults:
    '''
    Contain solution output of 1D simulation
    - Contains conservative Q Matrix (shape: (3, N)) at all time steps
    - Contains flux matrix (shape: (3, N+1)) at all time steps
    - Contains value of time at each iteration
    - Contains associated solutions config to aid in plotting/analysis

    Q_hist: list of ndarray [N_iter of np.array(3, N_cell)]
    F_hist: list of ndarray [N_iter of np.array(3, N_cell)]
    t_hist: list [N_iter]
    '''

    Q_hist: list
    F_hist: list
    t_hist: list

    config: EulerConfig


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_results(results: EulerResults, interval: int = 30, save_path: str = None):
    '''
    Animates Density, Velocity, and Pressure profiles over time.

    :param EulerResults results: Results dataclass from EulerSolver
    :param int interval: Delay between frames in milliseconds (default: 30ms)
    :param str save_path: Optional path to save as .mp4 or .gif (e.g. 'sod_shock.gif')
    '''
    # Extract mesh and time history
    N_cells = results.config.N_cells
    domain_size = results.config.domain_size
    dx = results.config.dx
    gamma = results.config.gamma

    # Cell center coordinates
    x = np.linspace(dx / 2.0, domain_size - dx / 2.0, N_cells)

    # Pre-calculate primitive variables for every frame: rho, u, P
    rho_hist, u_hist, p_hist = [], [], []

    for Q in results.Q_hist:
        rho = Q[0, :]
        u = Q[1, :] / rho
        P = (gamma - 1.0) * (Q[2, :] - 0.5 * rho * u**2)

        rho_hist.append(rho)
        u_hist.append(u)
        p_hist.append(P)

    # Set up figure and 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle('1D Euler Equation Dynamics', fontsize=14, fontweight='bold')

    # Line initializations
    line_rho, = ax1.plot([], [], 'b-', lw=2, label=r'Density ($\rho$)')
    line_u,   = ax2.plot([], [], 'r-', lw=2, label=r'Velocity ($u$)')
    line_p,   = ax3.plot([], [], 'g-', lw=2, label=r'Pressure ($P$)')

    # Configure axes limits
    for ax, label in zip([ax1, ax2, ax3], ['Density', 'Velocity', 'Pressure']):
        ax.set_xlim(0, domain_size)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    # Dynamic dynamic limits with padding
    rho_flat = np.concatenate(rho_hist)
    u_flat   = np.concatenate(u_hist)
    p_flat   = np.concatenate(p_hist)

    ax1.set_ylim(np.min(rho_flat) - 0.1, np.max(rho_flat) + 0.1)
    ax2.set_ylim(np.min(u_flat) - 0.5,   np.max(u_flat) + 0.5)
    ax3.set_ylim(np.min(p_flat) - 0.1,   np.max(p_flat) + 0.1)

    ax3.set_xlabel('x')
    time_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        line_rho.set_data([], [])
        line_u.set_data([], [])
        line_p.set_data([], [])
        time_text.set_text('')
        return line_rho, line_u, line_p, time_text

    def update(frame):
        t = results.t_hist[frame]
        
        line_rho.set_data(x, rho_hist[frame])
        line_u.set_data(x, u_hist[frame])
        line_p.set_data(x, p_hist[frame])
        
        time_text.set_text(f'Time: {t:.4f} s')
        return line_rho, line_u, line_p, time_text

    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(results.t_hist),
        init_func=init, 
        blit=True, 
        interval=interval
    )

    if save_path:
        anim.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'pillow')

    plt.tight_layout()
    plt.show()

    return anim