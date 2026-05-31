"""Render a short GIF preview without opening the interactive GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CG_LAB_NUM_PARTICLES", "2500")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import taichi as ti

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "demo.gif"
sys.path.insert(0, str(ROOT))

from src.Work0.config import BACKGROUND_COLOR, PARTICLE_COLOR, WINDOW_RES


def _hex_to_rgb(value: int) -> tuple[int, int, int]:
    return (value >> 16 & 255, value >> 8 & 255, value & 255)


def _mouse_path(frame: int, total: int) -> tuple[float, float]:
    t = frame / max(1, total - 1)
    return (
        0.5 + 0.34 * np.cos(2.0 * np.pi * t),
        0.5 + 0.28 * np.sin(4.0 * np.pi * t),
    )


def _draw_frame(points: np.ndarray, mouse: tuple[float, float]) -> Image.Image:
    width, height = WINDOW_RES
    bg = (246, 248, 252)
    particle = (0, 86, 179)
    glow_color = _hex_to_rgb(PARTICLE_COLOR)
    attractor = (220, 38, 38)

    image = Image.new("RGB", WINDOW_RES, bg)
    glow = Image.new("RGBA", WINDOW_RES, (0, 0, 0, 0))
    draw_glow = ImageDraw.Draw(glow, "RGBA")

    pixel_points = np.empty_like(points)
    pixel_points[:, 0] = points[:, 0] * width
    pixel_points[:, 1] = (1.0 - points[:, 1]) * height

    for x, y in pixel_points[::3]:
        draw_glow.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(*glow_color, 70))

    image = Image.alpha_composite(image.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(2)))
    draw = ImageDraw.Draw(image, "RGBA")

    for x, y in pixel_points:
        draw.ellipse((x - 1.4, y - 1.4, x + 1.4, y + 1.4), fill=(*particle, 210))

    mx, my = mouse
    cx, cy = mx * width, (1.0 - my) * height
    draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), outline=(*attractor, 240), width=3)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(*attractor, 255))
    return image.convert("RGB")


def main() -> None:
    ti.init(arch=ti.cpu)

    from src.Work0.physics import init_particles, pos, update_particles

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    init_particles()

    frames = []
    frame_count = 64
    steps_per_frame = 3

    for frame in range(frame_count):
        mouse = _mouse_path(frame, frame_count)
        for _ in range(steps_per_frame):
            update_particles(mouse[0], mouse[1])
        frames.append(_draw_frame(pos.to_numpy(), mouse))

    imageio.mimsave(OUTPUT, frames, duration=0.045, loop=0)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
