"""Experiment 3: interactive Bezier curve with De Casteljau algorithm."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


WIDTH = 800
HEIGHT = 800
BACKGROUND = 0xFFFFFF
POLYGON_COLOR = 0x9CA3AF
CURVE_COLOR = 0x16A34A
POINT_COLOR = 0xDC2626
MAX_POINTS = 100
SAMPLES = 1000


def de_casteljau(control_points: np.ndarray, t: float) -> np.ndarray:
    points = control_points.astype(np.float32).copy()
    n = len(points)
    for level in range(1, n):
        points[: n - level] = (1.0 - t) * points[: n - level] + t * points[1 : n - level + 1]
    return points[0]


def sample_curve(control_points: np.ndarray) -> np.ndarray:
    if len(control_points) < 2:
        return np.empty((0, 2), dtype=np.float32)
    ts = np.linspace(0.0, 1.0, SAMPLES + 1, dtype=np.float32)
    return np.array([de_casteljau(control_points, float(t)) for t in ts], dtype=np.float32)


def draw_scene(gui: ti.GUI, control_points: np.ndarray) -> None:
    gui.clear(BACKGROUND)

    if len(control_points) >= 2:
        for i in range(len(control_points) - 1):
            gui.line(tuple(control_points[i]), tuple(control_points[i + 1]), radius=2, color=POLYGON_COLOR)

        curve = sample_curve(control_points)
        for i in range(len(curve) - 1):
            gui.line(tuple(curve[i]), tuple(curve[i + 1]), radius=2, color=CURVE_COLOR)

    if len(control_points) > 0:
        gui.circles(control_points, radius=7, color=POINT_COLOR)


def _capture_frame(gui: ti.GUI) -> np.ndarray:
    frame = (np.clip(gui.get_image(), 0.0, 1.0) * 255).astype(np.uint8)
    return np.flipud(np.rot90(frame))


def record_gif(output: str | os.PathLike[str]) -> None:
    ti.init(arch=ti.cpu)
    gui = ti.GUI("Experiment 3: Bezier Curve", res=(WIDTH, HEIGHT), show_gui=False)
    target_points = np.array(
        [
            [0.10, 0.22],
            [0.22, 0.82],
            [0.44, 0.18],
            [0.66, 0.80],
            [0.88, 0.35],
        ],
        dtype=np.float32,
    )
    frames = []
    control_points = np.empty((0, 2), dtype=np.float32)
    for idx in range(len(target_points)):
        for alpha in np.linspace(0.0, 1.0, 12):
            if idx == 0:
                control_points = target_points[:1]
            else:
                moving = target_points[idx - 1] * (1.0 - alpha) + target_points[idx] * alpha
                control_points = np.vstack([target_points[:idx], moving]).astype(np.float32)
            draw_scene(gui, control_points)
            frames.append(_capture_frame(gui))
            gui.show()

    for _ in range(30):
        draw_scene(gui, target_points)
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
    gui = ti.GUI("Experiment 3: Bezier Curve", res=(WIDTH, HEIGHT))
    control_points = []
    print("Left click to add control points, press C to clear, Esc to exit.")

    while gui.running:
        for event in gui.get_events():
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == ti.GUI.LMB and event.type == ti.GUI.PRESS:
                if len(control_points) < MAX_POINTS:
                    control_points.append(gui.get_cursor_pos())
            elif event.key in ("c", "C") and event.type == ti.GUI.PRESS:
                control_points.clear()

        points = np.array(control_points, dtype=np.float32)
        draw_scene(gui, points)
        gui.show()


if __name__ == "__main__":
    run()

