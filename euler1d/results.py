'''
1D Euler Solver Results Hub

Stores Results from the 1D Euler Solver into EulerResults Class
Also contains visualizaton functions
'''

from dataclasses import dataclass, field
import numpy as np

@dataclass
class EulerResults:
    '''
    Contain solution output of 1D simulation
    -
    
    '''