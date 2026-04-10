"""Generate synthetic line images similar to data/real_data/line examples.

Each image is 576x576 with:
- A noisy scattered trail of dots along a random line direction
- Random outlier dots
- 0-2 random solid shapes (rectangles, circles, ellipses) as obstacles
"""

import os
import random
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "line")
NUM_IMAGES = 1000
IMG_SIZE = 576
DPI = 96
FIG_SIZE = IMG_SIZE / DPI


def random_line_params():
    angle = random.uniform(0, 2 * math.pi)
    cx = random.randint(150, IMG_SIZE - 150)
    cy = random.randint(150, IMG_SIZE - 150)
    length = random.randint(200, 450)
    return cx, cy, angle, length


def generate_dots(rng, cx, cy, angle, length):
    """Generate dot positions along a line with noise + outliers."""
    num_dots = rng.integers(150, 400)
    num_outliers = rng.integers(10, 40)

    dx = math.cos(angle)
    dy = math.sin(angle)
    base_perp = rng.uniform(5, 20)

    t = rng.normal(0, length / 3, num_dots)
    t = np.clip(t, -length, length)

    tail_factor = 1.0 + 0.3 * np.abs(t) / length
    perp = rng.normal(0, 1, num_dots) * base_perp * tail_factor

    x = cx + t * dx - perp * dy
    y = cy + t * dy + perp * dx

    # Outliers
    x_out = rng.uniform(20, IMG_SIZE - 20, num_outliers)
    y_out = rng.uniform(20, IMG_SIZE - 20, num_outliers)

    x = np.concatenate([x, x_out])
    y = np.concatenate([y, y_out])

    mask = (x >= 0) & (x < IMG_SIZE) & (y >= 0) & (y < IMG_SIZE)
    return x[mask], y[mask]


def render_dots(x, y):
    """Render dots using matplotlib scatter and return as PIL Image."""
    fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.scatter(x, y, s=7, c="black", linewidths=0)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)
    ax.invert_yaxis()
    ax.axis("off")

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    plt.close(fig)

    img = Image.fromarray(buf, "RGBA")
    return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def draw_random_shapes(draw):
    num_shapes = random.choices([0, 1, 2, 3], weights=[5, 3, 1, 1])[0]
    for _ in range(num_shapes):
        shape_type = random.choice(["rect", "ellipse", "circle"])
        x = random.randint(20, IMG_SIZE - 80)
        y = random.randint(20, IMG_SIZE - 80)

        if shape_type == "rect":
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            angle = random.uniform(-45, 45)
            temp = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            temp_draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, 255))
            temp = temp.rotate(angle, center=(x + w // 2, y + h // 2), expand=False)
            draw._image.paste(Image.alpha_composite(draw._image.copy(), temp))

        elif shape_type == "ellipse":
            w = random.randint(30, 90)
            h = random.randint(20, 70)
            draw.ellipse([x, y, x + w, y + h], fill=(0, 0, 0, 255))

        elif shape_type == "circle":
            r = random.randint(10, 40)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0, 255))


def generate_image(index, rng):
    cx, cy, angle, length = random_line_params()
    x, y = generate_dots(rng, cx, cy, angle, length)

    img = render_dots(x, y)
    draw = ImageDraw.Draw(img)
    draw_random_shapes(draw)

    path = os.path.join(OUTPUT_DIR, f"line_{index:05d}.png")
    img.save(path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    rng = np.random.default_rng(42)

    for i in range(NUM_IMAGES):
        generate_image(i, rng)

    print(f"Generated {NUM_IMAGES} synthetic line images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
