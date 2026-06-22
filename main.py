import numpy as np

from canvas import (
    create_canvas,
    generate_quadrilateral,
    centroid_of_polygon,
    draw_center,
    distribute_centers_in_quadrilateral,
)
from smoke import draw_smoke
from trajectories import draw_trajectories
from export import tensor_to_image

HEIGHT = 720
WIDTH = 1280


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    tensor = create_canvas(height, width)

    quad = generate_quadrilateral(height, width, rng)
    origin = centroid_of_polygon(quad)

    center_size = rng.integers(1, 3)
    draw_center(tensor, origin, center_size)

    num_centers = rng.integers(40, 80)
    centers = distribute_centers_in_quadrilateral(quad, num_centers, rng)

    for center in centers:
        draw_center(tensor, center, rng.integers(0, 2))

    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    smoke_radius = base_length * rng.uniform(0.15, 0.3)
    draw_smoke(tensor, centers, smoke_radius, rng)

    num_straight = rng.integers(1, 16)
    num_parabolic = rng.integers(1, 16)
    draw_trajectories(tensor, centers, origin, num_straight, num_parabolic, rng)

    tensor = np.where(tensor >= 128, 255, 0).astype(np.uint8)

    return tensor


def main():
    for i in range(1, 5):
        tensor = generate_explosion(HEIGHT, WIDTH)
        path = f"explosion_{i}.png"
        tensor_to_image(tensor, path)


if __name__ == "__main__":
    main()
