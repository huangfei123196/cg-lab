"""Taichi fields and GPU kernels for the gravity particle swarm."""

import taichi as ti

from .config import (
    BOUNCE_COEF,
    DRAG_COEF,
    GRAVITY_STRENGTH,
    INITIAL_SPEED,
    MAX_ACCELERATION,
    NUM_PARTICLES,
    SOFTENING,
)


pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)


@ti.kernel
def init_particles():
    """Initialize particle positions and small random velocities."""
    for i in range(NUM_PARTICLES):
        pos[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        vel[i] = ti.Vector(
            [
                (ti.random(ti.f32) - 0.5) * INITIAL_SPEED,
                (ti.random(ti.f32) - 0.5) * INITIAL_SPEED,
            ]
        )


@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    """Update all particles in parallel using a softened inverse-square pull."""
    attractor = ti.Vector([mouse_x, mouse_y])

    for i in range(NUM_PARTICLES):
        direction = attractor - pos[i]
        dist2 = direction.dot(direction) + SOFTENING
        dist = ti.sqrt(dist2)
        acceleration = direction / dist * (GRAVITY_STRENGTH / dist2)

        accel_norm = acceleration.norm()
        if accel_norm > MAX_ACCELERATION:
            acceleration = acceleration / accel_norm * MAX_ACCELERATION

        vel[i] = (vel[i] + acceleration) * DRAG_COEF
        pos[i] += vel[i]

        for axis in ti.static(range(2)):
            if pos[i][axis] < 0.0:
                pos[i][axis] = 0.0
                vel[i][axis] *= BOUNCE_COEF
            elif pos[i][axis] > 1.0:
                pos[i][axis] = 1.0
                vel[i][axis] *= BOUNCE_COEF
