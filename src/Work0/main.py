"""GUI entry point for the Taichi gravity particle swarm."""

from __future__ import annotations

import taichi as ti

from .config import BACKGROUND_COLOR, PARTICLE_COLOR, PARTICLE_RADIUS, WINDOW_RES


def _init_taichi() -> None:
    try:
        ti.init(arch=ti.gpu)
    except Exception as exc:
        print(f"GPU backend is unavailable, falling back to CPU: {exc}")
        ti.init(arch=ti.cpu)


def run() -> None:
    _init_taichi()

    from .physics import init_particles, pos, update_particles

    print("Compiling Taichi kernels, please wait...")
    init_particles()

    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("Ready. Move the mouse inside the window to attract the particles.")

    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        update_particles(mouse_x, mouse_y)
        gui.clear(BACKGROUND_COLOR)
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()


if __name__ == "__main__":
    run()

