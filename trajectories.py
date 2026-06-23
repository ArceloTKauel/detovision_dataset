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

    pixels_since_draw = 0
    next_draw_at = 0
    max_spacing = 50

    for py, px in points:
        if pixels_since_draw >= next_draw_at:
            if 0 <= py < h and 0 <= px < w:
                tensor[py, px] = 255
            dist_from_origin = np.sqrt((py - oy) ** 2 + (px - ox) ** 2)
            max_dist = length if length > 0 else 1
            ratio = min(dist_from_origin / max_dist, 1.0)
            spacing = ratio ** 2 * max_spacing
            next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
            pixels_since_draw = 0
        else:
            pixels_since_draw += 1


def draw_parabolic_trajectory(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle: float,
    length: float,
    curvature: float,
    origin: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center
    oy, ox = origin

    perp_angle = angle + np.pi / 2
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cos_p = np.cos(perp_angle)
    sin_p = np.sin(perp_angle)

    num_steps = int(length)
    if num_steps < 2:
        return

    prev_py, prev_px = cy, cx
    pixels_since_draw = 0
    next_draw_at = 0
    max_spacing = 50

    for i in range(num_steps):
        t = i / num_steps * length
        offset = curvature * t * t

        py = int(round(cy + t * sin_a + offset * sin_p))
        px = int(round(cx + t * cos_a + offset * cos_p))

        if i > 0 and (abs(py - prev_py) > 1 or abs(px - prev_px) > 1):
            segment = bresenham(prev_py, prev_px, py, px)
        else:
            segment = [(py, px)]

        for sy, sx in segment:
            if pixels_since_draw >= next_draw_at:
                if 0 <= sy < h and 0 <= sx < w:
                    tensor[sy, sx] = 255
                dist_from_origin = np.sqrt((sy - oy) ** 2 + (sx - ox) ** 2)
                max_dist = length if length > 0 else 1
                ratio = min(dist_from_origin / max_dist, 1.0)
                spacing = ratio ** 2 * max_spacing
                next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
                pixels_since_draw = 0
            else:
                pixels_since_draw += 1

        prev_py, prev_px = py, px


def draw_straight_trajectories(
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


def draw_parabolic_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    drone_angle: float,
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

        curvature = rng.uniform(-0.005, 0.005)
        if abs(curvature) < 0.001:
            curvature = 0.001 * (1 if rng.random() > 0.5 else -1)

        angle_diff = angle - drone_angle
        curvature_factor = abs(np.sin(angle_diff))
        curvature *= curvature_factor

        draw_parabolic_trajectory(tensor, center, angle, length, curvature, origin, rng)


def draw_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_straight: int,
    num_parabolic: int,
    drone_angle: float,
    rng: np.random.Generator,
) -> None:
    draw_straight_trajectories(tensor, centers, origin, num_straight, rng)
    draw_parabolic_trajectories(tensor, centers, origin, num_parabolic, drone_angle, rng)
