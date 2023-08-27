"""
The forward model. Functions and classes related to rotation and
projection, involving tri-linear and nearest-neighbour interpolation,
as well as jit-compiled functions useful for simulating images and
reconstruction.
"""

__all__ = ["interpolate", "project_spatial", "project", "apply_shifts_and_ctf", "rotate_and_interpolate", "Slice"]

from .interpolation import interpolate
from .projection import project_spatial, project, apply_shifts_and_ctf, rotate_and_interpolate
from .slice import Slice
