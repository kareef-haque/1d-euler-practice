'''
Generates dataset for ML Training
- Solutions of randomly generated versions of a shock tube problem
- Will automatically save and create a folder named 'generated_data' in the current working directory
    - Generates a Train/Test/Validation Split
        - Default Ratio is 80/20/0
            - User may modify it here
- Saves outputs as .npy files
    - Outputs are numerically solved Q at every timestep (N_iter, 3, N_cells)
        Q = Conservative State Matrix: Shape (3, N_cells)
- Also saves <name>_meta.npz next to each trajectory (initial states, interface position, dx, dt,
  t_hist, gamma) so exact solutions can be reconstructed and euler1d/neural/dataset.py can
  coarse-grain the data consistently
'''

import numpy as np
import os
from run import random_Run
from infrastructure.solver_config import cons_to_prim

# Dataset configuration parameters defined here
    #NOTE: MODIFY random_Run() IN run.py TO MODIFY SOLVER PARAMETERS
 
N_samples = 20

percent_train = 0.8
percent_test = 0.2
percent_validate = 0.0

# Save path (DO NOT MODIFY)
data_path = os.path.join(os.getcwd(), "generated_data")
train_save_path = os.path.join(data_path, "train")
test_save_path = os.path.join(data_path, "test")
validate_save_path = os.path.join(data_path, "validate")


def _save_sample(save_dir, name):
    result, physics = random_Run(return_physics=True)
    cfg = result.config
    np.save(os.path.join(save_dir, f"{name}.npy"), np.array(result.Q_hist))
    np.savez(os.path.join(save_dir, f"{name}_meta.npz"),
             prim_L=cons_to_prim(physics.Q_L[:, None], cfg.gamma)[:, 0],
             prim_R=cons_to_prim(physics.Q_R[:, None], cfg.gamma)[:, 0],
             x_interface=physics.domain_split_percent * cfg.domain_size,
             domain_size=cfg.domain_size, dx=cfg.dx, dt=cfg.dt,
             t_hist=np.array(result.t_hist), gamma=cfg.gamma)


def generate_Euler_Dataset():

    # Create data storage folders if missing
    for path in [data_path, train_save_path, test_save_path, validate_save_path]:
        os.makedirs(path, exist_ok=True)

    splits = [("train", train_save_path, int(N_samples*percent_train)),
              ("test", test_save_path, int(N_samples*percent_test)),
              ("validate", validate_save_path, int(N_samples*percent_validate))]

    for split, save_dir, n in splits:
        for i in range(n):
            print('============================================')
            print(f"Generating {split} data {i+1}/{n}")
            print('============================================')
            _save_sample(save_dir, f"{split}_{i}")


if __name__ == '__main__':
    generate_Euler_Dataset()
