import numpy as np

from smoke import measure_smoke_width


def bresenham(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def draw_trajectory(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle: float,
    length: float,
    origin: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center
    oy, ox = origin

    end_y = int(round(cy + length * np.sin(angle)))
    end_x = int(round(cx + length * np.cos(angle)))

    end_y = np.clip(end_y, 0, h - 1)
    end_x = np.clip(end_x, 0, w - 1)

    points = bresenham(cy, cx, end_y, end_x)

    for py, px in points:
        dist_from_origin = np.sqrt((py - oy) ** 2 + (px - ox) ** 2)
        max_dist = length if length > 0 else 1
        ratio = min(dist_from_origin / max_dist, 1.0)

        gap_probability = ratio * 0.85

        if rng.random() > gap_probability:
            if 0 <= py < h and 0 <= px < w:
                tensor[py, px] = 255


def draw_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)

    for _ in range(num_trajectories):
        center = centers[rng.integers(0, len(centers))]
        angle = rng.uniform(0, 2 * np.pi)

        smoke_width = measure_smoke_width(tensor, origin, angle)
        min_length = max(10, smoke_width)
        length = rng.uniform(min_length, diagonal)

        draw_trajectory(tensor, center, angle, length, origin, rng)
