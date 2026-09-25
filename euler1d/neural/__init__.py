'''Neural finite-volume (NFV-style) components for the 1D Euler solver. Requires PyTorch.'''
from .adapter import NeuralFluxScheme
from .model import NeuralEulerFlux, load_neural_flux, save_neural_flux
from .stepper import ClassicalFlux, rollout, step

__all__ = ['NeuralEulerFlux', 'NeuralFluxScheme', 'ClassicalFlux', 'load_neural_flux',
           'save_neural_flux', 'rollout', 'step']
