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


def random_point_in_quadrilateral(vertices: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    v0, v1, v2, v3 = vertices
    if rng.random() < 0.5:
        tri = [v0, v1, v2]
    else:
        tri = [v0, v2, v3]
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    point = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
    return int(round(point[0])), int(round(point[1]))


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    tensor = create_canvas(height, width)

    quad = generate_quadrilateral(height, width, rng)
    num_centers = rng.integers(2, 5)

    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    cut_angle = rng.uniform(-0.17, 0.17)

    for _ in range(num_centers):
        center = random_point_in_quadrilateral(quad, rng)

        center_size = rng.integers(1, 3)
        draw_center(tensor, center, center_size)

        cut_spread = rng.uniform(15, 25)
        cut_length = base_length * rng.uniform(0.5, 1.0)
        cut_trajectories = rng.integers(30, 70)

        draw_lobe(tensor, center, cut_angle + rng.uniform(-0.1, 0.1), cut_spread, cut_length, cut_trajectories, rng)
        draw_lobe(tensor, center, cut_angle + np.pi + rng.uniform(-0.1, 0.1), cut_spread, cut_length * rng.uniform(0.5, 1.0), cut_trajectories, rng)

        explosion_angle = cut_angle + np.pi / 2 + rng.uniform(-0.17, 0.17)
        explosion_spread = rng.uniform(20, 40)
        explosion_length = base_length * rng.uniform(0.8, 1.4)
        explosion_trajectories = rng.integers(40, 100)

        draw_lobe(tensor, center, explosion_angle, explosion_spread, explosion_length, explosion_trajectories, rng)

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
