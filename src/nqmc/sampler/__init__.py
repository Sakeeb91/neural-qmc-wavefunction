"""MCMC samplers for VMC.

This module provides Markov Chain Monte Carlo samplers for generating
electron configurations distributed according to |ψ|².

Example:
    >>> from nqmc.sampler import MetropolisSampler
    >>> sampler = MetropolisSampler(step_size=0.1)
"""
from nqmc.sampler.metropolis import MetropolisSampler, SamplerState

__all__ = [
    "MetropolisSampler",
    "SamplerState",
]
