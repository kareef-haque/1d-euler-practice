# 1D Euler Solver for ML Studies

1D Euler equation solver devloped by Kareef and Pranet for use in research at the Visual Intelligence Laboratory (UVA). Solver will be used for experiments in machine learning, with a focus on integrating ML into CFD while justifying this integration
- Using Flux-Reconstruction Schema
- Shared Wrapper that runs the solver
- Each developer coded up their own flux-reconstruction method
- Both methods combinations successfully validated against exact solution of the Sod Shock Tube problem

Kareef: AUSM + & WENO 5

Pranet: HLLC & WENO 5Z


# ML Experiments & Questions:

### Kareef:

### Pranet:
Using INR (Implicit Neural Representation) to provide a continuous state representation for the 1D Euler problem. 
- Analysing how well INR behaves around the discontinuties in the flow
    - Augmentations to improve INR representation of shocks
- Attempting to map how the neural net weights & biases map to the flow field
    - How does the change in NN weights/biases affect the flow field, and can this be used as a latent space representation?
- How does differents INR formulations (INR, SIREN, WIRE) compare?
    - When given the same training, how do their weights & biases vary?
        - How does the solution quality vary, and what does the differences in weights & biases say about how weights & biases affect the flow field?


Key Elements of FVM Method [Flux & Reconstruction]
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
- pytorch




