"""Central configuration for the particle simulation."""

from __future__ import annotations

import os


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


# Physics parameters
NUM_PARTICLES = _int_from_env("CG_LAB_NUM_PARTICLES", 10_000)
GRAVITY_STRENGTH = 0.00008
SOFTENING = 0.0025
MAX_ACCELERATION = 0.003
DRAG_COEF = 0.985
BOUNCE_COEF = -0.8
INITIAL_SPEED = 0.002

# Render parameters
WINDOW_RES = (800, 600)
PARTICLE_RADIUS = 1.5
PARTICLE_COLOR = 0x00BFFF
BACKGROUND_COLOR = 0x050816

