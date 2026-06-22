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

    explosion_line_length = base_length * rng.uniform(0.3, 0.5)
    explosion_num_centers = rng.integers(5, 12)
    explosion_centers = distribute_centers_along_line(origin, explosion_angle, explosion_line_length, explosion_num_centers, rng)

    for center in explosion_centers:
        draw_center(tensor, center, rng.integers(0, 2))

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
