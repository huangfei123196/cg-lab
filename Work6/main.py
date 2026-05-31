"""Experiment 6 low-difficulty version: differentiable rendering in Taichi."""

from __future__ import annotations

import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


RES = 256
TARGET_LIGHT = (0.8, 0.8, 0.2)
SPHERE_CENTER = ti.Vector([0.5, 0.5, 0.5])
SPHERE_RADIUS = 0.3
ti.init(arch=ti.gpu)

target_pixels = ti.field(dtype=ti.f32, shape=(RES, RES))
current_pixels = ti.field(dtype=ti.f32, shape=(RES, RES))
display_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES * 2, RES))
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def generate_target():
    target = ti.Vector(TARGET_LIGHT)
    for i, j in target_pixels:
        x = (i + 0.5) / RES
        y = (j + 0.5) / RES
        dx = x - SPHERE_CENTER[0]
        dy = y - SPHERE_CENTER[1]
        dist_sq = dx * dx + dy * dy
        value = 0.0
        if dist_sq < SPHERE_RADIUS * SPHERE_RADIUS:
            z = SPHERE_CENTER[2] - ti.sqrt(SPHERE_RADIUS * SPHERE_RADIUS - dist_sq)
            p = ti.Vector([x, y, z])
            n = (p - SPHERE_CENTER).normalized()
            l_dir = (target - p).normalized()
            value = ti.max(0.0, n.dot(l_dir))
        target_pixels[i, j] = value


@ti.kernel
def render_and_compute_loss():
    for i, j in target_pixels:
        x = (i + 0.5) / RES
        y = (j + 0.5) / RES
        dx = x - SPHERE_CENTER[0]
        dy = y - SPHERE_CENTER[1]
        dist_sq = dx * dx + dy * dy
        intensity = 0.0

        if dist_sq < SPHERE_RADIUS * SPHERE_RADIUS:
            z = SPHERE_CENTER[2] - ti.sqrt(SPHERE_RADIUS * SPHERE_RADIUS - dist_sq)
            p = ti.Vector([x, y, z])
            n = (p - SPHERE_CENTER).normalized()
            l_dir = (light_pos[None] - p).normalized()
            dot_value = n.dot(l_dir)
            intensity = ti.max(0.1 * dot_value, dot_value)

        diff = intensity - target_pixels[i, j]
        loss[None] += diff * diff / (RES * RES)
        current_pixels[i, j] = ti.max(0.0, ti.min(1.0, intensity))


@ti.kernel
def build_display():
    for i, j in target_pixels:
        t = target_pixels[i, j]
        c = current_pixels[i, j]
        display_pixels[i, j] = ti.Vector([t, t, t])
        display_pixels[i + RES, j] = ti.Vector([c, c, c])


def _display_frame() -> np.ndarray:
    image = (display_pixels.to_numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.flipud(np.rot90(image))


def optimize(record_frames: bool = False) -> list[np.ndarray]:
    light_pos[None] = [0.2, 0.2, 0.8]
    m = np.zeros(3, dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    beta1, beta2 = 0.9, 0.999
    lr, eps = 0.025, 1e-8
    frames: list[np.ndarray] = []

    for iteration in range(1, 241):
        loss[None] = 0.0
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()

        grad = np.array(light_pos.grad[None], dtype=np.float32)
        for c in range(3):
            m[c] = beta1 * m[c] + (1.0 - beta1) * grad[c]
            v[c] = beta2 * v[c] + (1.0 - beta2) * grad[c] * grad[c]
            m_hat = m[c] / (1.0 - beta1**iteration)
            v_hat = v[c] / (1.0 - beta2**iteration)
            light_pos[None][c] -= lr * m_hat / (math.sqrt(float(v_hat)) + eps)

        if iteration % 10 == 0:
            pos = light_pos[None]
            print(
                f"iter={iteration:03d} loss={loss[None]:.6f} "
                f"light=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )

        if record_frames and (iteration <= 80 or iteration % 3 == 0):
            build_display()
            frames.append(_display_frame())

    build_display()
    if record_frames:
        for _ in range(24):
            frames.append(_display_frame())
    return frames


def record_gif(output: str | os.PathLike[str]) -> None:
    generate_target()
    frames = optimize(record_frames=True)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.045, loop=0)
    print(f"Wrote {output_path}")


def run() -> None:
    gif_output = os.getenv("CG_LAB_RECORD_GIF")
    if gif_output:
        record_gif(gif_output)
        return

    generate_target()
    window = ti.GUI("Differentiable Rendering (Left: Target, Right: Current)", res=(RES * 2, RES))
    light_pos[None] = [0.2, 0.2, 0.8]
    m = np.zeros(3, dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    beta1, beta2 = 0.9, 0.999
    lr, eps = 0.025, 1e-8
    iteration = 0

    while window.running and iteration < 1000:
        iteration += 1
        loss[None] = 0.0
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()
        grad = np.array(light_pos.grad[None], dtype=np.float32)
        for c in range(3):
            m[c] = beta1 * m[c] + (1.0 - beta1) * grad[c]
            v[c] = beta2 * v[c] + (1.0 - beta2) * grad[c] * grad[c]
            m_hat = m[c] / (1.0 - beta1**iteration)
            v_hat = v[c] / (1.0 - beta2**iteration)
            light_pos[None][c] -= lr * m_hat / (math.sqrt(float(v_hat)) + eps)
        build_display()
        window.set_image(display_pixels)
        window.show()


if __name__ == "__main__":
    run()
