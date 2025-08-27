"""
Reinforcement Learning module for JANI environments.

This module provides:
- Training scripts for RL agents on JANI environments
- Simulation and evaluation of learned policies
- Integration with stable-baselines3 and sb3-contrib
"""

from .simulation import Simulator

__all__ = ['Simulator']