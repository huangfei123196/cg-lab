"""Experiment 2: rotation and MVP transformation."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


WIDTH = 700
HEIGHT = 700
BACKGROUND = 0xF7F9FC
TRIANGLE_COLOR = 0x1D4ED8
VERTEX_COLOR = 0xDC2626

TRIANGLE_VERTICES = np.array(
    [
        [1.4, 0.0, -4.0],
        [-0.8, 1.2, -4.0],
        [-0.8, -1.2, -4.0],
    ],
    dtype=np.float32,
)


def get_model_matrix(angle_deg: float) -> np.ndarray:
    """Return a model matrix that rotates around the Z axis."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_view_matrix(eye_pos: np.ndarray) -> np.ndarray:
    """Move the world so the camera sits at the origin."""
    view = np.eye(4, dtype=np.float32)
    view[:3, 3] = -eye_pos[:3]
    return view


def get_projection_matrix(
    eye_fov: float, aspect_ratio: float, z_near: float, z_far: float
) -> np.ndarray:
    """Build a perspective projection matrix with right-handed camera space."""
    fov = np.radians(eye_fov)
    n = -z_near
    f = -z_far
    top = np.tan(fov / 2.0) * abs(n)
    bottom = -top
    right = aspect_ratio * top
    left = -right

    persp_to_ortho = np.array(
        [
            [n, 0.0, 0.0, 0.0],
            [0.0, n, 0.0, 0.0],
            [0.0, 0.0, n + f, -n * f],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ortho_scale = np.array(
        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, 2.0 / (n - f), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ortho_translate = np.array(
        [
            [1.0, 0.0, 0.0, -(right + left) / 2.0],
            [0.0, 1.0, 0.0, -(top + bottom) / 2.0],
            [0.0, 0.0, 1.0, -(n + f) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ortho_scale @ ortho_translate @ persp_to_ortho


def transform_vertices(angle_deg: float) -> np.ndarray:
    model = get_model_matrix(angle_deg)
    view = get_view_matrix(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    projection = get_projection_matrix(45.0, WIDTH / HEIGHT, 0.1, 50.0)
    mvp = projection @ view @ model

    points = []
    for vertex in TRIANGLE_VERTICES:
        homogeneous = np.array([vertex[0], vertex[1], vertex[2], 1.0], dtype=np.float32)
        clip = mvp @ homogeneous
        ndc = clip[:3] / clip[3]
        x = (ndc[0] + 1.0) * 0.5
        y = (ndc[1] + 1.0) * 0.5
        points.append([x, y])
    return np.array(points, dtype=np.float32)


def draw_triangle(gui: ti.GUI, points: np.ndarray) -> None:
    gui.clear(BACKGROUND)
    for i in range(3):
        a = points[i]
        b = points[(i + 1) % 3]
        gui.line(tuple(a), tuple(b), radius=3, color=TRIANGLE_COLOR)
    gui.circles(points, radius=7, color=VERTEX_COLOR)


def _capture_frame(gui: ti.GUI) -> np.ndarray:
    frame = (np.clip(gui.get_image(), 0.0, 1.0) * 255).astype(np.uint8)
    return np.flipud(np.rot90(frame))


def record_gif(output: str | os.PathLike[str]) -> None:
    ti.init(arch=ti.cpu)
    gui = ti.GUI("Experiment 2: Rotation and Transformation", res=(WIDTH, HEIGHT), show_gui=False)
    frames = []
    for angle in np.linspace(0.0, 360.0, 90, endpoint=False):
        draw_triangle(gui, transform_vertices(angle))
        frames.append(_capture_frame(gui))
        gui.show()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.04, loop=0)
    print(f"Wrote {output_path}")


def run() -> None:
    gif_output = os.getenv("CG_LAB_RECORD_GIF")
    if gif_output:
        record_gif(gif_output)
        return

    ti.init(arch=ti.cpu)
    gui = ti.GUI("Experiment 2: Rotation and Transformation", res=(WIDTH, HEIGHT))
    angle = 0.0
    print("Press A/D to rotate the triangle, Esc to exit.")

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key in ("a", "A"):
                angle += 10.0
            elif event.key in ("d", "D"):
                angle -= 10.0

        draw_triangle(gui, transform_vertices(angle))
        gui.show()


if __name__ == "__main__":
    run()

