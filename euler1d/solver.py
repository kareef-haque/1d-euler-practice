'''
1D Euler Equation Solver

- Two Flux Reconstruction Methods
    - Kareef: AUSM + & WENO 5
    - Pranet: HLLC & WENO 5Z

Specify simulation parameters via ...
Specify solver parameters via...
'''

import numpy as np
from boundary import apply_BC
from prob_config import EulerConfig
from results import EulerResults





def EulerSolver(config: EulerConfig,
                flux = 'WENO5Z',
                reconstruction = 'HLLC'
                ):
    '''
    :param EulerConfig config: Configuration for the problem
    :param str flux: Flux Scheme (WENO5Z, WENO5)
    :param str reconstruction: Reconstruction Method (AUSM+, HLLC)

    :return EulerResults: Results of the simulation
    

    Main function to run the 1D Euler simulation
        - integrates RK4 for time stepping
    '''
    

    return 