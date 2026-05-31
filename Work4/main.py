"""Experiment 4: Phong illumination with ray intersections."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti


WIDTH = 512
HEIGHT = 512
ti.init(arch=ti.gpu)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

ka = ti.field(dtype=ti.f32, shape=())
kd = ti.field(dtype=ti.f32, shape=())
ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())


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
        if t0 > 1e-4:
            t = t0
        elif t1 > 1e-4:
            t = t1
        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    height = apex.y - base_y
    k = (radius / height) ** 2
    ro_local = ro - apex
    a = rd.x**2 + rd.z**2 - k * rd.y**2
    b = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    c = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    if ti.abs(a) > 1e-6:
        disc = b * b - 4.0 * a * c
        if disc > 0.0:
            root = ti.sqrt(disc)
            t0 = (-b - root) / (2.0 * a)
            t1 = (-b + root) / (2.0 * a)
            if t0 > t1:
                t0, t1 = t1, t0
            y0 = ro_local.y + t0 * rd.y
            y1 = ro_local.y + t1 * rd.y
            if t0 > 1e-4 and -height <= y0 <= 0.0:
                t = t0
            elif t1 > 1e-4 and -height <= y1 <= 0.0:
                t = t1
            if t > 0.0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
    return t, normal


@ti.func
def phong(p, n, base_color, eye):
    light_pos = ti.Vector([2.5, 3.5, 4.0])
    light_color = ti.Vector([1.0, 1.0, 1.0])
    ambient = ka[None] * base_color
    l = normalize(light_pos - p)
    v = normalize(eye - p)
    r = normalize(2.0 * n.dot(l) * n - l)
    diff = kd[None] * ti.max(n.dot(l), 0.0) * base_color
    spec = ks[None] * ti.pow(ti.max(r.dot(v), 0.0), shininess[None]) * light_color
    return ti.math.clamp(ambient + diff + spec, 0.0, 1.0)


@ti.kernel
def render():
    eye = ti.Vector([0.0, 0.0, 5.0])
    for i, j in pixels:
        u = (i - WIDTH * 0.5) / HEIGHT * 2.0
        v = (j - HEIGHT * 0.5) / HEIGHT * 2.0
        ro = eye
        rd = normalize(ti.Vector([u, v, -1.5]))

        min_t = 1e9
        hit_n = ti.Vector([0.0, 0.0, 0.0])
        hit_c = ti.Vector([0.0, 0.0, 0.0])

        t_sphere, n_sphere = intersect_sphere(ro, rd, ti.Vector([-0.95, -0.1, 0.0]), 1.0)
        if 0.0 < t_sphere < min_t:
            min_t = t_sphere
            hit_n = n_sphere
            hit_c = ti.Vector([0.85, 0.18, 0.14])

        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.05, 1.1, 0.0]), -1.25, 1.0)
        if 0.0 < t_cone < min_t:
            min_t = t_cone
            hit_n = n_cone
            hit_c = ti.Vector([0.52, 0.23, 0.85])

        color = ti.Vector([0.05, 0.09, 0.14]) + 0.18 * ti.Vector([u + 1.0, v + 1.0, 1.0])
        if min_t < 1e8:
            p = ro + rd * min_t
            color = phong(p, hit_n, hit_c, eye)
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


def _frame() -> np.ndarray:
    return np.flipud(np.rot90((pixels.to_numpy() * 255).clip(0, 255).astype(np.uint8)))


def record_gif(output: str | os.PathLike[str]) -> None:
    frames = []
    for t in np.linspace(0.0, 1.0, 72):
        ka[None] = 0.15 + 0.10 * np.sin(t * np.pi)
        kd[None] = 0.55 + 0.25 * t
        ks[None] = 0.25 + 0.45 * t
        shininess[None] = 12.0 + 84.0 * t
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

    ka[None], kd[None], ks[None], shininess[None] = 0.2, 0.7, 0.5, 32.0
    window = ti.ui.Window("Experiment 4: Phong Illumination", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    gui = window.get_gui()
    while window.running:
        with gui.sub_window("Phong Parameters", 10, 10, 260, 170):
            ka[None] = gui.slider_float("Ka", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)
        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    run()
