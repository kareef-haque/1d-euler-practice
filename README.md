# Practice 1D Euler Solver

1D Euler equation solver devloped by Kareef and Pranet as practice for research at the Visual Intelligence Laboratory (UVA). Potential integration with a physics-aware ML (PAML) model.

- Using Flux-Reconstruction Schema
- Shared Wrapper that runs the solver
- Each developer coded up their own flux-reconstruction method

Kareef: AUSM + & WENO 5

Pranet: HLLC & WENO 5Z
Notes on Pranet I/O:
- Reconstruction: 
    - Input: Conservative State Matrix (Q), dx (optional)
        - Q is shape (3, N+6), with N cells and 6 ghost cells (3 cells at beginning and end)
        - Params = Q, dx = 1.0
    - Ouput: Reconstructed Left/Right interface state matrices
        - (QL, QR), both of shape (3, N+1)
        - Returns = (QL, QR)
- Flux Scheme
    - Input: Reconstructed Left/Right interface state matrices (QL, QR), dx (optional), gamma (optional)
        - (QL, QR), both of shape (3, N+1)
        - params = QL, QR, dx = 1.0, gamma = 1.4
    - Ouput: Flux matrix (F) for all interfaces
        - F is shape (3, N+1)
        - Returns = F
- assumed uniform cell size

Required Packages:
- numpy
- ...



