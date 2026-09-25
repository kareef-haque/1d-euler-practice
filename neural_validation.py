'''
Sod shock tube validation of a trained neural flux, run end-to-end through EulerSolver
(the same entry point as the classical schemes), compared against the exact solution
and against classical schemes at the SAME coarse resolution.

    python neural_validation.py --model runs/nfv_euler/model_best.pt [--plot sod_neural.png]
'''
import argparse

import numpy as np

from euler1d.neural.adapter import NeuralFluxScheme
from euler1d.neural.dataset import exact_cell_averages, sod_spec
from euler1d.solver import EulerSolver
from infrastructure.solver_config import EulerConfig, cons_to_prim


def run_case(flux, recon, N, dx, dt, t_max, IC, gamma, scheme=None):
    cfg = EulerConfig(domain_size=N * dx, N_cells=N, IC=IC.copy(), BC='Zero-Gradient',
                      t_max=t_max, dt=dt, gamma=gamma, N_ghost=3)
    res = EulerSolver(cfg, flux=flux, reconstruction=recon, neural_scheme=scheme,
                      time_integrator='SSPRK3', verbose=False)   # same integrator for every scheme
    return res.Q_hist[-1], res.t_hist[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--t_max', type=float, default=5e-4)
    ap.add_argument('--domain', type=float, default=1.0)
    ap.add_argument('--plot', default=None)
    args = ap.parse_args()

    scheme = NeuralFluxScheme(args.model)
    dx, dt, gamma = scheme.model.dx, scheme.model.dt, scheme.model.gamma
    N = int(round(args.domain / dx))
    n_steps = int(round(args.t_max / dt))
    t_max = n_steps * dt
    spec = sod_spec(0.5)

    full0 = exact_cell_averages(spec, N, 1, dx, dt, gamma, n_ghost=0)[0]
    exact = exact_cell_averages(spec, N, 2, dx, t_max, gamma, n_ghost=0)[1]

    cases = {
        'Neural': ('Neural', None, scheme),
        'HLLC+FirstOrder': ('HLLC', 'FirstOrder', None),
        'HLLC+WENO5': ('HLLC', 'WENO5', None),
        'HLLC+WENO5Z': ('HLLC', 'WENO5Z', None),
        'AUSM+ +WENO5': ('AUSM+', 'WENO5', None),
    }
    Pe = cons_to_prim(exact, gamma)
    scales = np.array([Pe[0].max(), np.sqrt(gamma * Pe[2].max() / Pe[0].max()), Pe[2].max()])
    print(f'Sod, N={N}, dx={dx:g}, dt={dt:g}, t={t_max:g} ({n_steps} steps)')
    print(f"{'scheme':<18}{'L1 rho':>12}{'L1 u':>12}{'L1 p':>12}{'mean':>12}")
    out = {}
    for name, (flux, recon, sch) in cases.items():
        Q, t_end = run_case(flux, recon, N, dx, dt, t_max, full0, gamma, sch)
        assert abs(t_end - t_max) < 1e-12 * max(1.0, t_max) + 1e-15
        P = cons_to_prim(Q, gamma)
        err = np.nanmean(np.abs(P - Pe), axis=1) / scales if np.all(np.isfinite(P)) else np.full(3, np.inf)
        out[name] = P
        print(f'{name:<18}' + ''.join(f'{e:>12.3e}' for e in err) + f'{err.mean():>12.3e}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = (np.arange(N) + 0.5) * dx
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for k, lab in enumerate(['Density', 'Velocity', 'Pressure']):
            axes[k].plot(x, Pe[k], 'k-', lw=2, label='Exact (cell avg)')
            for name, P in out.items():
                axes[k].plot(x, P[k], '-', lw=1, label=name)
            axes[k].set_ylabel(lab)
            axes[k].grid(alpha=0.4)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('x (m)')
        fig.suptitle(f'Sod shock tube, N={N}, t={t_max:.2e}s')
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f'saved {args.plot}')


if __name__ == '__main__':
    main()
