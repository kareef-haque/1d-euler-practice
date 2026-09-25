'''
NFV training of a neural numerical flux for the 1D Euler equations.

Run from the repository root:
    python -m euler1d.neural.train --logdir runs/nfv_euler

Methodology carried over from NFV (nathanlct/nfv, train.py):
  * the flux network is placed inside the finite-volume update and the whole rollout is unrolled,
    loss = mean over steps of || Q_pred(t) - Q_true(t) ||^2  (normalised per channel)
  * curriculum: dx, dt fixed; the window (nx) and horizon (nt) grow stage by stage
  * boundaries pinned to ground truth (ghost cells from the exact solution)
  * gradient-norm clipping; divergence recovery (NFV restarts, here: roll back + halve lr)
  * evaluation = L1/L2 errors and win-rate against a classical scheme on held-out problems
'''
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .dataset import (build_exact_dataset, load_numerical_dataset, max_wave_speed,
                      sample_riemann_specs, state_scales)
from .model import DTYPES, NeuralEulerFlux, save_neural_flux
from .stepper import ClassicalFlux, rollout, step

DEFAULT_SCHEDULE = json.dumps([
    {"epochs": 300, "lr": 1e-3, "nx": 40, "nt": 10},
    {"epochs": 300, "lr": 5e-4, "nx": 60, "nt": 25},
    {"epochs": 400, "lr": 2e-4, "nx": 100, "nt": 50},
    {"epochs": 300, "lr": 1e-4, "nx": 100, "nt": 100},   # reach the evaluation horizon
])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logdir', default='runs/nfv_euler')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = torch default)')
    # model
    p.add_argument('--stencil', type=int, default=4)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--act', default='ELU')
    p.add_argument('--baseline', default='hllc', choices=['hllc', 'rusanov', 'central'])
    p.add_argument('--reconstruction', default='first_order', choices=['first_order', 'weno5'])
    p.add_argument('--no_symmetrize', action='store_true')
    p.add_argument('--max_correction', type=float, default=0.5)
    p.add_argument('--no_positivity_guard', action='store_true')
    p.add_argument('--checkpoint', default=None, help='resume from a saved neural flux')
    # physics / discretisation
    p.add_argument('--gamma', type=float, default=1.4)
    p.add_argument('--dx', type=float, default=1e-2, help='fixed across stages (NFV convention)')
    p.add_argument('--dt', type=float, default=5e-6, help='default gives CFL ~0.6 for the PhysicsConfig distribution')
    p.add_argument('--integrator', default='ssprk3', choices=['euler', 'ssprk3', 'rk4'],
                   help='ssprk3 recommended: the positivity guard is only guaranteed for SSP integrators')
    p.add_argument('--max_cfl', type=float, default=0.9)
    # data
    p.add_argument('--source', default='exact', choices=['exact', 'numerical'])
    p.add_argument('--n_train', type=int, default=128)
    p.add_argument('--n_eval', type=int, default=32)
    p.add_argument('--subsamples', type=int, default=8, help='sub-cell points for exact cell averages')
    p.add_argument('--data_dir', default='generated_data')
    p.add_argument('--coarsen', type=int, default=10, help='[numerical] spatial coarsening factor')
    p.add_argument('--time_stride', type=int, default=3, help='[numerical] temporal stride')
    # training
    p.add_argument('--schedule', default=DEFAULT_SCHEDULE, help='JSON list of {epochs, lr, nx, nt}')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--loss', default='l2', choices=['l1', 'l2'])
    p.add_argument('--grad_norm_clip', type=float, default=1.0)
    p.add_argument('--no_grad_checkpoint', action='store_true', help='faster, but memory grows with nt')
    p.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    # eval
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--eval_nx', type=int, default=100)
    p.add_argument('--eval_nt', type=int, default=200)
    return p.parse_args(argv)


# losses / metrics
def unrolled_loss(flux_op, Q_BT, ghosts_BT, dt, dx, gamma, integrator, loss='l2', grad_checkpoint=True):
    '''
    Autoregressive rollout from Q_BT[:,0] with the loss accumulated at every step (NFV).
    With grad_checkpoint, activations of each time step are recomputed in the backward pass:
    memory O(nt * state) instead of O(nt * 4 RK stages * network activations), ~2x compute.
    '''
    scale = state_scales(Q_BT[:, 0], gamma)
    Q, total = Q_BT[:, 0], 0.0
    T = Q_BT.shape[1]

    def one_step(q, g):
        return step(q, dt, dx, flux_op, ghosts=g, integrator=integrator)

    for t in range(T - 1):
        g_t = ghosts_BT[:, t] if ghosts_BT is not None else None
        if grad_checkpoint and torch.is_grad_enabled():
            Q = checkpoint(one_step, Q, g_t, use_reentrant=False)
        else:
            Q = one_step(Q, g_t)
        d = (Q - Q_BT[:, t + 1]) / scale
        total = total + (d.abs().mean() if loss == 'l1' else d.square().mean())
    return total / (T - 1)


@torch.no_grad()
def rollout_errors(flux_op, Q_BT, ghosts_BT, dt, dx, gamma, integrator):
    '''Per-sample normalised L1/L2 errors over the whole rollout (t >= 1). Diverged -> inf.'''
    pred = rollout(Q_BT[:, 0], Q_BT.shape[1] - 1, dt, dx, flux_op, ghosts_BT=ghosts_BT, integrator=integrator)
    d = (pred[:, 1:] - Q_BT[:, 1:]) / state_scales(Q_BT[:, 0], gamma).unsqueeze(1)
    l1, l2 = d.abs().mean(dim=(1, 2, 3)), d.square().mean(dim=(1, 2, 3))
    bad = ~torch.isfinite(l2)
    l1[bad], l2[bad] = math.inf, math.inf
    return {'l1': l1.cpu(), 'l2': l2.cpu(), 'pred': pred}


def compare_schemes(schemes, Q_BT, ghosts_BT, dt, dx, gamma, integrator, reference='hllc_fo'):
    rows = {}
    for name, op in schemes.items():
        e = rollout_errors(op, Q_BT, ghosts_BT, dt, dx, gamma, integrator)
        rows[name] = {'l1': e['l1'], 'l2': e['l2']}
    out = {}
    for name, e in rows.items():
        ref = rows[reference]
        out[name] = {
            'l1_mean': float(e['l1'].mean()), 'l2_mean': float(e['l2'].mean()),
            f'winrate_l2_vs_{reference}': float((e['l2'] < ref['l2']).double().mean()),
            'l2_ratio_vs_ref': float(ref['l2'].mean() / e['l2'].mean()),
        }
    return out


def print_table(res):
    names = list(res)
    keys = list(res[names[0]])
    print('  ' + f"{'scheme':<14}" + ''.join(f'{k:>26}' for k in keys))
    for n in names:
        print('  ' + f'{n:<14}' + ''.join(f'{res[n][k]:>26.4e}' for k in keys))


# data
def make_data(args, specs, nx, nt, device):
    if args.source == 'exact':
        return build_exact_dataset(specs, nx, nt, args.dx, args.dt, args.gamma, 3, args.subsamples,
                                   dtype=DTYPES[args.dtype], device=device)
    Q, ghosts, meta = load_numerical_dataset(args.data_dir, args.coarsen, args.time_stride,
                                             max_steps=nt - 1, dtype=DTYPES[args.dtype], device=device)
    if meta['dx'] and (abs(meta['dx'] / args.dx - 1) > 1e-6 or abs(meta['dt'] / args.dt - 1) > 1e-6):
        raise ValueError(f"numerical data gives dx={meta['dx']}, dt={meta['dt']}; pass matching --dx/--dt")
    return Q, ghosts


# training
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    schedule = json.loads(args.schedule)
    dev, gamma = args.device, args.gamma

    if args.checkpoint:
        from .model import load_neural_flux
        model = load_neural_flux(args.checkpoint, dev).train()
    else:
        model = NeuralEulerFlux(stencil=args.stencil, hidden=args.hidden, depth=args.depth, act=args.act,
                                baseline=args.baseline, reconstruction=args.reconstruction,
                                symmetrize=not args.no_symmetrize, max_correction=args.max_correction,
                                positivity_guard=not args.no_positivity_guard, integrator=args.integrator,
                                gamma=gamma, dx=args.dx, dt=args.dt, dtype=args.dtype).to(dev)
    print(f'NeuralEulerFlux: {model.num_params()} parameters, config={model.config}')

    # held-out evaluation problems (different seed from training)
    train_specs = sample_riemann_specs(args.n_train, seed=args.seed + 1)
    eval_specs = sample_riemann_specs(args.n_eval, seed=args.seed + 10_000)
    print(f'Generating eval data (nx={args.eval_nx}, nt={args.eval_nt}, N={args.n_eval})...')
    if args.source == 'exact':
        Q_ev, G_ev = build_exact_dataset(eval_specs, args.eval_nx, args.eval_nt, args.dx, args.dt, gamma, 3,
                                         args.subsamples, dtype=DTYPES[args.dtype], device=dev)
    else:
        Q_all, G_ev = make_data(args, None, None, args.eval_nt, dev)
        Q_ev = Q_all[-args.n_eval:]
    cfl = max_wave_speed(Q_ev, gamma) * args.dt / args.dx
    print(f'  max CFL on eval data = {cfl:.3f}')
    if cfl > args.max_cfl:
        raise ValueError(f'CFL {cfl:.2f} > {args.max_cfl}: reduce --dt')

    baselines = {
        'hllc_fo': ClassicalFlux('first_order', 'hllc', gamma, dx=args.dx),
        'rusanov_fo': ClassicalFlux('first_order', 'rusanov', gamma, dx=args.dx),
        'hllc_weno5': ClassicalFlux('weno5', 'hllc', gamma, dx=args.dx),
    }
    base_res = compare_schemes(baselines, Q_ev, G_ev, args.dt, args.dx, gamma, args.integrator)
    print('Baselines on eval set:')
    print_table(base_res)
    ref_l2 = rollout_errors(baselines['hllc_fo'], Q_ev, G_ev, args.dt, args.dx, gamma, args.integrator)['l2']

    history, best = [], (math.inf, None)
    metrics_f = open(os.path.join(args.logdir, 'metrics.jsonl'), 'w')
    total_epochs = 0
    for s_idx, stage in enumerate(schedule):
        print(f"\n=== stage {s_idx}: {stage}")
        t0 = time.time()
        if args.source == 'exact':
            Q_tr, G_tr = make_data(args, train_specs, stage['nx'], stage['nt'], dev)
        else:
            Q_all, G_tr = make_data(args, None, None, stage['nt'], dev)
            Q_tr = Q_all[:-args.n_eval] if len(Q_all) > args.n_eval else Q_all
        print(f'  train data {tuple(Q_tr.shape)} in {time.time() - t0:.1f}s, '
              f'CFL={max_wave_speed(Q_tr, gamma) * args.dt / args.dx:.3f}')

        lr = stage['lr']
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        last_good = copy.deepcopy(model.state_dict())
        t0 = time.time()
        for epoch in range(stage['epochs']):
            idx = torch.randperm(len(Q_tr))[:args.batch_size]
            loss = unrolled_loss(model, Q_tr[idx], G_tr[idx] if G_tr is not None else None,
                                 args.dt, args.dx, gamma, args.integrator, args.loss,
                                 grad_checkpoint=not args.no_grad_checkpoint)
            opt.zero_grad()
            if not torch.isfinite(loss):
                model.load_state_dict(last_good)
                lr *= 0.5
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                print(f'  epoch {epoch}: non-finite loss -> rolled back, lr={lr:.1e}')
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip))
            if not math.isfinite(gn):
                opt.zero_grad()
                print(f'  epoch {epoch}: non-finite gradient -> step skipped')
                continue
            opt.step()
            last_good = copy.deepcopy(model.state_dict())
            total_epochs += 1

            if epoch % args.eval_every == 0 or epoch == stage['epochs'] - 1:
                model.eval()
                e = rollout_errors(model, Q_ev, G_ev, args.dt, args.dx, gamma, args.integrator)
                model.train()
                ref = base_res['hllc_fo']
                l2 = float(e['l2'].mean())
                wr = float((e['l2'] < ref_l2).double().mean())
                rec = dict(stage=s_idx, epoch=epoch, total_epochs=total_epochs, loss=float(loss), grad_norm=gn,
                           eval_l1=float(e['l1'].mean()), eval_l2=l2, winrate_l2_vs_hllc_fo=wr,
                           l2_ratio_vs_hllc_fo=ref['l2_mean'] / l2, eps=(epoch + 1) / (time.time() - t0))
                history.append(rec)
                metrics_f.write(json.dumps(rec) + '\n')
                metrics_f.flush()
                print(f"  ep {epoch:5d} | loss {rec['loss']:.3e} | gn {gn:.2e} | eval l2 {l2:.3e} "
                      f"| improv. vs HLLC-FO x{rec['l2_ratio_vs_hllc_fo']:.2f} | winrate {100 * wr:.0f}% "
                      f"| {rec['eps']:.1f} ep/s")
                if l2 < best[0]:
                    best = (l2, total_epochs)
                    save_neural_flux(model, os.path.join(args.logdir, 'model_best.pt'),
                                     extra={'eval_l2': l2, 'total_epochs': total_epochs})
        save_neural_flux(model, os.path.join(args.logdir, f'model_stage{s_idx}.pt'))

    save_neural_flux(model, os.path.join(args.logdir, 'model_final.pt'))
    metrics_f.close()

    from .model import load_neural_flux
    best_model = load_neural_flux(os.path.join(args.logdir, 'model_best.pt'), dev)
    final = compare_schemes({**baselines, 'neural': best_model}, Q_ev, G_ev, args.dt, args.dx, gamma,
                            args.integrator)
    print(f'\nFinal comparison on held-out problems (best model @ epoch {best[1]}):')
    print_table(final)
    with open(os.path.join(args.logdir, 'final_eval.json'), 'w') as f:
        json.dump(final, f, indent=2)
    return best_model, final


if __name__ == '__main__':
    train(parse_args())
