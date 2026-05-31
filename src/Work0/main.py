"""GUI entry point for the Taichi gravity particle swarm."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti

from .config import BACKGROUND_COLOR, PARTICLE_COLOR, PARTICLE_RADIUS, WINDOW_RES


def _init_taichi() -> None:
    try:
        ti.init(arch=ti.gpu)
    except Exception as exc:
        print(f"GPU backend is unavailable, falling back to CPU: {exc}")
        ti.init(arch=ti.cpu)


def _mouse_path(frame: int, total: int) -> tuple[float, float]:
    t = frame / max(1, total - 1)
    return (
        0.5 + 0.34 * np.cos(2.0 * np.pi * t),
        0.5 + 0.28 * np.sin(4.0 * np.pi * t),
    )


def _draw_frame(gui: ti.GUI, mouse_x: float, mouse_y: float) -> None:
    from .physics import pos, update_particles

    update_particles(mouse_x, mouse_y)
    gui.clear(BACKGROUND_COLOR)
    gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
    gui.circle((mouse_x, mouse_y), color=0xFF4040, radius=8)


def record_gif(output_path: str | os.PathLike[str], frame_count: int = 72) -> None:
    _init_taichi()

    from .physics import init_particles

    init_particles()
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES, show_gui=False)
    frames = []

    for frame in range(frame_count):
        mouse_x, mouse_y = _mouse_path(frame, frame_count)
        _draw_frame(gui, mouse_x, mouse_y)
        image = (np.clip(gui.get_image(), 0.0, 1.0) * 255).astype(np.uint8)
        image = np.flipud(np.rot90(image))
        frames.append(image)
        gui.show()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=0.045, loop=0)
    print(f"Wrote {output}")


def run() -> None:
    gif_output = os.getenv("CG_LAB_RECORD_GIF")
    if gif_output:
        record_gif(gif_output)
        return

    _init_taichi()

    from .physics import init_particles

    print("Compiling Taichi kernels, please wait...")
    init_particles()

    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("Ready. Move the mouse inside the window to attract the particles.")

    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        _draw_frame(gui, mouse_x, mouse_y)
        gui.show()


if __name__ == "__main__":
    run()
