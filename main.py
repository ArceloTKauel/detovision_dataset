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
    lateral_spread: float = 3.0,
) -> list[tuple[int, int]]:
    cy, cx = origin
    centers = []
    for _ in range(num_points):
        t = rng.uniform(-line_length / 2, line_length / 2)
        offset = rng.uniform(-lateral_spread, lateral_spread)
        py = int(round(cy + t * np.sin(angle) + offset * np.cos(angle)))
        px = int(round(cx + t * np.cos(angle) - offset * np.sin(angle)))
        centers.append((py, px))
    return centers


def draw_smoke(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    smoke_radius: float,
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape

    all_cy = np.array([c[0] for c in centers], dtype=np.float64)
    all_cx = np.array([c[1] for c in centers], dtype=np.float64)

    min_y = max(0, int(all_cy.min() - smoke_radius - 10))
    max_y = min(h, int(all_cy.max() + smoke_radius + 10))
    min_x = max(0, int(all_cx.min() - smoke_radius - 10))
    max_x = min(w, int(all_cx.max() + smoke_radius + 10))

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    min_dist = np.full(ys.shape, np.inf)
    for cy, cx in centers:
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        min_dist = np.minimum(min_dist, dist)

    core_radius = smoke_radius * 0.25
    core_mask = min_dist < core_radius
    tensor[min_y:max_y, min_x:max_x][core_mask] = 255

    mid_radius = smoke_radius * 0.5
    mid_mask = (min_dist >= core_radius) & (min_dist < mid_radius)
    ratio_mid = (min_dist[mid_mask] - core_radius) / (mid_radius - core_radius)
    noise_mid = rng.random(np.count_nonzero(mid_mask))
    prob_mid = 0.9 - 0.4 * ratio_mid
    region = tensor[min_y:max_y, min_x:max_x]
    brightness_mid = (255 * (1 - ratio_mid * 0.3)).astype(np.uint8)
    region[mid_mask] = np.where(
        noise_mid < prob_mid,
        np.maximum(region[mid_mask], brightness_mid),
        region[mid_mask],
    )

    outer_mask = (min_dist >= mid_radius) & (min_dist < smoke_radius)
    ratio_outer = (min_dist[outer_mask] - mid_radius) / (smoke_radius - mid_radius)
    noise_outer = rng.random(np.count_nonzero(outer_mask))
    prob_outer = 0.6 * (1 - ratio_outer) ** 2
    brightness_outer = (200 * (1 - ratio_outer * 0.7)).astype(np.uint8)
    region[outer_mask] = np.where(
        noise_outer < prob_outer,
        np.maximum(region[outer_mask], brightness_outer),
        region[outer_mask],
    )

    fringe_radius = smoke_radius * 1.3
    fringe_mask = (min_dist >= smoke_radius) & (min_dist < fringe_radius)
    ratio_fringe = (min_dist[fringe_mask] - smoke_radius) / (fringe_radius - smoke_radius)
    noise_fringe = rng.random(np.count_nonzero(fringe_mask))
    prob_fringe = 0.15 * (1 - ratio_fringe) ** 3
    brightness_fringe = (120 * (1 - ratio_fringe)).astype(np.uint8)
    region[fringe_mask] = np.where(
        noise_fringe < prob_fringe,
        np.maximum(region[fringe_mask], brightness_fringe),
        region[fringe_mask],
    )


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

    cut_line_length = base_length * rng.uniform(0.5, 0.8)
    cut_num_centers = rng.integers(30, 60)
    cut_centers = distribute_centers_along_line(origin, cut_angle, cut_line_length, cut_num_centers, rng, lateral_spread=5.0)

    for center in cut_centers:
        draw_center(tensor, center, rng.integers(0, 2))

    cut_smoke_radius = base_length * rng.uniform(0.15, 0.3)
    draw_smoke(tensor, cut_centers, cut_smoke_radius, rng)

    explosion_line_length = base_length * rng.uniform(0.4, 0.7)
    explosion_num_centers = rng.integers(30, 60)
    explosion_centers = distribute_centers_along_line(origin, explosion_angle, explosion_line_length, explosion_num_centers, rng, lateral_spread=5.0)

    for center in explosion_centers:
        draw_center(tensor, center, rng.integers(0, 2))

    explosion_smoke_radius = base_length * rng.uniform(0.2, 0.35)
    draw_smoke(tensor, explosion_centers, explosion_smoke_radius, rng)

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
