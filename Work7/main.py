"""Experiment 7: mass-spring cloth with multiple integrators."""

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


N = 20
NUM_PARTICLES = N * N
MAX_SPRINGS = 2 * N * (N - 1)
DT = 0.003
MASS = 1.0
K_S = 650.0
K_D = 2.8
MAX_VELOCITY = 6.0
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
ti.init(arch=ti.gpu)

x = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
x_tmp = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v_tmp = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
is_fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)
spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
num_springs = ti.field(dtype=ti.i32, shape=())
screen_points = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.475, 0.65, j * 0.05 - 0.475])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[idx] = 1 if j == 0 and (i == 0 or i == N - 1) else 0


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            c = ti.atomic_add(num_springs[None], 1)
            other = (i + 1) * N + j
            spring_pairs[c] = ti.Vector([idx, other])
            spring_lengths[c] = (x[idx] - x[other]).norm()
        if j < N - 1:
            c = ti.atomic_add(num_springs[None], 1)
            other = i * N + (j + 1)
            spring_pairs[c] = ti.Vector([idx, other])
            spring_lengths[c] = (x[idx] - x[other]).norm()


def init_cloth() -> None:
    num_springs[None] = 0
    init_positions()
    init_springs()


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(NUM_PARTICLES):
        force[i] = GRAVITY * MASS - K_D * vel[i]
    for s in range(num_springs[None]):
        a = spring_pairs[s][0]
        b = spring_pairs[s][1]
        delta = pos[a] - pos[b]
        dist = delta.norm()
        if dist > 1e-6:
            direction = delta / dist
            spring_force = -K_S * (dist - spring_lengths[s]) * direction
            ti.atomic_add(force[a], spring_force)
            ti.atomic_add(force[b], -spring_force)


@ti.func
def clamp_velocity(field: ti.template(), idx: int):
    speed = field[idx].norm()
    if speed > MAX_VELOCITY:
        field[idx] = field[idx] / speed * MAX_VELOCITY


@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] += v[i] * DT
            v[i] += f[i] / MASS * DT
            clamp_velocity(v, i)


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v[i] += f[i] / MASS * DT
            clamp_velocity(v, i)
            x[i] += v[i] * DT


@ti.kernel
def copy_state():
    for i in range(NUM_PARTICLES):
        x_tmp[i] = x[i]
        v_tmp[i] = v[i]


@ti.kernel
def implicit_iteration():
    compute_forces_on(x_tmp, v_tmp, f)
    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v_tmp[i] = v[i] + f[i] / MASS * DT
            clamp_velocity(v_tmp, i)
            x_tmp[i] = x[i] + v_tmp[i] * DT


@ti.kernel
def commit_tmp():
    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] = x_tmp[i]
            v[i] = v_tmp[i]


def step_implicit_iter() -> None:
    copy_state()
    for _ in range(4):
        implicit_iteration()
    commit_tmp()


@ti.kernel
def update_screen_points():
    for i in range(NUM_PARTICLES):
        p = x[i]
        screen_points[i] = ti.Vector([0.5 + p.x * 0.78 + p.z * 0.18, 0.18 + p.y * 0.72 - p.z * 0.10])


def draw_cloth(gui: ti.GUI, method_name: str) -> None:
    gui.clear(0xF7F9FC)
    update_screen_points()
    points = screen_points.to_numpy()
    pairs = spring_pairs.to_numpy()[: num_springs[None]]
    for a, b in pairs:
        gui.line(tuple(points[a]), tuple(points[b]), radius=1, color=0x64748B)
    gui.circles(points, radius=3, color=0x2563EB)
    gui.text(f"Integrator: {method_name}", pos=(0.03, 0.95), color=0x111827, font_size=20)


def _capture_frame(gui: ti.GUI) -> np.ndarray:
    image = np.nan_to_num(gui.get_image(), nan=0.0, posinf=1.0, neginf=0.0)
    frame = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return np.flipud(np.rot90(frame))


def record_gif(output: str | os.PathLike[str]) -> None:
    init_cloth()
    gui = ti.GUI("Experiment 7: Mass Spring Cloth", res=(800, 800), show_gui=False)
    frames = []
    phases = [
        ("Explicit", step_explicit, 70),
        ("Semi-Implicit", step_semi_implicit, 70),
        ("Implicit Iteration", step_implicit_iter, 90),
    ]
    for name, step, count in phases:
        init_cloth()
        for frame in range(count):
            for _ in range(6):
                step()
            draw_cloth(gui, name)
            if frame % 2 == 0:
                frames.append(_capture_frame(gui))
            gui.show()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.045, loop=0)
    print(f"Wrote {output_path}")


def run() -> None:
    gif_output = os.getenv("CG_LAB_RECORD_GIF")
    if gif_output:
        record_gif(gif_output)
        return

    init_cloth()
    gui = ti.GUI("Experiment 7: Mass Spring Cloth", res=(800, 800))
    method = 1
    paused = False
    print("Press 1/2/3 to switch integrators, Space to pause, R to reset, Esc to exit.")
    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == "1":
                method = 0
            elif event.key == "2":
                method = 1
            elif event.key == "3":
                method = 2
            elif event.key == ti.GUI.SPACE:
                paused = not paused
            elif event.key in ("r", "R"):
                init_cloth()

        name = ["Explicit", "Semi-Implicit", "Implicit Iteration"][method]
        if not paused:
            for _ in range(8):
                if method == 0:
                    step_explicit()
                elif method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()
        draw_cloth(gui, name)
        gui.show()


if __name__ == "__main__":
    run()
