import numpy as np
from PIL import Image

HEIGHT = 720
WIDTH = 1280


def create_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def generate_quadrilateral(
    height: int, width: int, rng: np.random.Generator, margin: float = 0.15
) -> np.ndarray:
    cx = rng.integers(int(width * margin), int(width * (1 - margin)))
    cy = rng.integers(int(height * margin), int(height * (1 - margin)))

    angles = np.sort(rng.uniform(0, 2 * np.pi, size=4))
    radii = rng.uniform(20, 60, size=4)

    vertices = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        vertices[i, 0] = cy + radii[i] * np.sin(angles[i])
        vertices[i, 1] = cx + radii[i] * np.cos(angles[i])

    return vertices


def centroid_of_polygon(vertices: np.ndarray) -> tuple[int, int]:
    cy = int(round(np.mean(vertices[:, 0])))
    cx = int(round(np.mean(vertices[:, 1])))
    return cy, cx


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
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center

    end_y = int(round(cy + length * np.sin(angle)))
    end_x = int(round(cx + length * np.cos(angle)))

    end_y = np.clip(end_y, 0, h - 1)
    end_x = np.clip(end_x, 0, w - 1)

    points = bresenham(cy, cx, end_y, end_x)

    for i, (py, px) in enumerate(points):
        dist_from_center = np.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        max_dist = length if length > 0 else 1
        ratio = dist_from_center / max_dist

        gap_probability = ratio * 0.8

        if rng.random() > gap_probability:
            if 0 <= py < h and 0 <= px < w:
                brightness = int(255 * (1 - ratio * 0.7))
                tensor[py, px] = max(tensor[py, px], brightness)


def draw_lobe(
    tensor: np.ndarray,
    center: tuple[int, int],
    base_angle: float,
    spread_deg: float,
    max_length: float,
    num_trajectories: int,
    rng: np.random.Generator,
) -> None:
    spread_rad = np.radians(spread_deg)

    for _ in range(num_trajectories):
        angle = base_angle + rng.uniform(-spread_rad, spread_rad)
        length = max_length * rng.uniform(0.3, 1.0)
        draw_trajectory(tensor, center, angle, length, rng)


def draw_center(
    tensor: np.ndarray,
    center: tuple[int, int],
    size: int,
) -> None:
    h, w = tensor.shape
    cy, cx = center
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                tensor[ny, nx] = 255


def distribute_centers_along_line(
    origin: tuple[int, int],
    angle: float,
    line_length: float,
    num_points: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    cy, cx = origin
    centers = []
    for _ in range(num_points):
        t = rng.uniform(-line_length / 2, line_length / 2)
        offset = rng.uniform(-3, 3)
        py = int(round(cy + t * np.sin(angle) + offset * np.cos(angle)))
        px = int(round(cx + t * np.cos(angle) - offset * np.sin(angle)))
        centers.append((py, px))
    return centers


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    tensor = create_canvas(height, width)

    quad = generate_quadrilateral(height, width, rng)
    origin = centroid_of_polygon(quad)

    center_size = rng.integers(1, 3)
    draw_center(tensor, origin, center_size)

    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    cut_angle = rng.uniform(-0.17, 0.17)
    explosion_angle = cut_angle + np.pi / 2 + rng.uniform(-0.17, 0.17)

    cut_line_length = base_length * rng.uniform(0.3, 0.6)
    cut_num_centers = rng.integers(5, 12)
    cut_centers = distribute_centers_along_line(origin, cut_angle, cut_line_length, cut_num_centers, rng)

    for center in cut_centers:
        draw_center(tensor, center, rng.integers(0, 2))

        spread = rng.uniform(10, 20)
        length = base_length * rng.uniform(0.4, 0.9)
        num_traj = rng.integers(15, 35)

        draw_lobe(tensor, center, cut_angle + rng.uniform(-0.1, 0.1), spread, length, num_traj, rng)
        draw_lobe(tensor, center, cut_angle + np.pi + rng.uniform(-0.1, 0.1), spread, length * rng.uniform(0.5, 1.0), num_traj, rng)

    explosion_line_length = base_length * rng.uniform(0.3, 0.5)
    explosion_num_centers = rng.integers(5, 12)
    explosion_centers = distribute_centers_along_line(origin, explosion_angle, explosion_line_length, explosion_num_centers, rng)

    for center in explosion_centers:
        draw_center(tensor, center, rng.integers(0, 2))

        spread = rng.uniform(15, 35)
        length = base_length * rng.uniform(0.6, 1.3)
        num_traj = rng.integers(20, 45)

        draw_lobe(tensor, center, explosion_angle + rng.uniform(-0.15, 0.15), spread, length, num_traj, rng)

    return tensor


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)
    print(f"Imagen guardada en: {path}")


def main():
    for i in range(1, 5):
        tensor = generate_explosion(HEIGHT, WIDTH)
        path = f"explosion_{i}.jpg"
        tensor_to_image(tensor, path)


if __name__ == "__main__":
    main()
