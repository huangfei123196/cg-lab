"""Experiment 5: iterative ray tracing with reflections and shadows."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


WIDTH = 512
HEIGHT = 512
MAT_DIFFUSE = 0
MAT_MIRROR = 1
EPS = 1e-4
ti.init(arch=ti.gpu)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())


@ti.func
def normalize(v):
    return v / ti.max(v.norm(), 1e-6)


@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    disc = b * b - c
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    if disc > 0.0:
        s = ti.sqrt(disc)
        t0 = -b - s
        t1 = -b + s
        if t0 > EPS:
            t = t0
        elif t1 > EPS:
            t = t1
        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_plane(ro, rd, y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-6:
        candidate = (y - ro.y) / rd.y
        if candidate > EPS:
            t = candidate
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e9
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.15, 0.0, 0.0]), 0.9)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.85, 0.12, 0.1])
        hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([1.15, 0.0, -0.15]), 0.9)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.92, 0.92, 0.95])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -0.9)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        p = ro + rd * t
        cell = (ti.floor(p.x * 1.6) + ti.floor(p.z * 1.6)) % 2
        if cell == 0:
            hit_c = ti.Vector([0.82, 0.82, 0.78])
        else:
            hit_c = ti.Vector([0.28, 0.30, 0.34])
        hit_mat = MAT_DIFFUSE

    return min_t, hit_n, hit_c, hit_mat


@ti.func
def visible_to_light(p, n, light):
    l = light - p
    dist = l.norm()
    l_dir = l / ti.max(dist, 1e-6)
    shadow_t, _, _, _ = scene_intersect(p + n * EPS, l_dir)
    visible = 1.0
    if 0.0 < shadow_t < dist:
        visible = 0.0
    return visible


@ti.kernel
def render():
    light = light_pos[None]
    for i, j in pixels:
        u = (i - WIDTH * 0.5) / HEIGHT * 2.0
        v = (j - HEIGHT * 0.5) / HEIGHT * 2.0
        ro = ti.Vector([0.0, 0.75, 4.5])
        rd = normalize(ti.Vector([u, v - 0.25, -1.5]))
        final = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        bg = ti.Vector([0.05, 0.09, 0.13])

        for _ in range(max_bounces[None]):
            t, n, obj_color, mat_id = scene_intersect(ro, rd)
            if t > 1e8:
                final += throughput * bg
                break

            p = ro + rd * t
            if mat_id == MAT_MIRROR:
                rd = normalize(rd - 2.0 * rd.dot(n) * n)
                ro = p + n * EPS
                throughput *= 0.82 * obj_color
            else:
                l = normalize(light - p)
                diff = ti.max(n.dot(l), 0.0)
                shadow = visible_to_light(p, n, light)
                ambient = 0.12 * obj_color
                final += throughput * (ambient + shadow * diff * obj_color)
                break

        pixels[i, j] = ti.math.clamp(final, 0.0, 1.0)


def _frame() -> np.ndarray:
    return np.flipud(np.rot90((pixels.to_numpy() * 255).clip(0, 255).astype(np.uint8)))


def record_gif(output: str | os.PathLike[str]) -> None:
    max_bounces[None] = 3
    frames = []
    for t in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
        light_pos[None] = [2.5 * np.cos(t), 3.2, 2.5 + 1.5 * np.sin(t)]
        render()
        frames.append(_frame())
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.045, loop=0)
    print(f"Wrote {output_path}")


def run() -> None:
    gif_output = os.getenv("CG_LAB_RECORD_GIF")
    if gif_output:
        record_gif(gif_output)
        return

    light_pos[None] = [2.2, 3.0, 3.0]
    max_bounces[None] = 3
    window = ti.ui.Window("Experiment 5: Ray Tracing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    gui = window.get_gui()
    while window.running:
        with gui.sub_window("Ray Tracing Parameters", 10, 10, 280, 170):
            lp = light_pos[None]
            lp[0] = gui.slider_float("Light X", lp[0], -4.0, 4.0)
            lp[1] = gui.slider_float("Light Y", lp[1], 0.5, 5.0)
            lp[2] = gui.slider_float("Light Z", lp[2], -1.0, 6.0)
            light_pos[None] = lp
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    run()
