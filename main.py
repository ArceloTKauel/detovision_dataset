import numpy as np
from PIL import Image

HEIGHT = 720
WIDTH = 1280


def create_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def random_center(height: int, width: int, margin: float = 0.15) -> tuple[int, int]:
    rng = np.random.default_rng()
    cy = rng.integers(int(height * margin), int(height * (1 - margin)))
    cx = rng.integers(int(width * margin), int(width * (1 - margin)))
    return int(cy), int(cx)


def draw_lobe(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle_deg: float,
    length: float,
    spread_deg: float,
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center
    angle_rad = np.radians(angle_deg)
    spread_rad = np.radians(spread_deg)

    ys, xs = np.mgrid[0:h, 0:w]
    dy = ys - cy
    dx = xs - cx
    r = np.sqrt(dy**2 + dx**2)
    theta = np.arctan2(dy, dx)

    angle_diff = np.abs(np.arctan2(np.sin(theta - angle_rad), np.cos(theta - angle_rad)))
    in_lobe = angle_diff < spread_rad

    nucleus_radius = length * 0.25
    nucleus_mask = in_lobe & (r < nucleus_radius)
    nucleus_prob = np.where(nucleus_mask, 1.0 - (r / nucleus_radius) * 0.3, 0.0)
    noise = rng.random(tensor.shape)
    tensor[nucleus_mask & (noise < nucleus_prob)] = 255

    mid_radius = length * 0.55
    mid_mask = in_lobe & (r >= nucleus_radius) & (r < mid_radius)
    mid_prob = np.where(mid_mask, 0.7 - (r / mid_radius) * 0.5, 0.0)
    tensor[mid_mask & (noise < mid_prob)] = 255

    num_filaments = rng.integers(15, 35)
    for _ in range(num_filaments):
        fil_angle = angle_rad + rng.uniform(-spread_rad * 0.9, spread_rad * 0.9)
        fil_length = length * rng.uniform(0.5, 1.0)
        fil_width = rng.uniform(0.8, 2.5)

        num_points = int(fil_length)
        t = np.linspace(0, fil_length, num_points)
        wobble = np.cumsum(rng.normal(0, 0.3, num_points))

        fil_x = cx + t * np.cos(fil_angle) + wobble * np.cos(fil_angle + np.pi / 2)
        fil_y = cy + t * np.sin(fil_angle) + wobble * np.sin(fil_angle + np.pi / 2)

        for i in range(num_points):
            px, py = int(round(fil_x[i])), int(round(fil_y[i]))
            brightness = max(0, 255 - int(255 * (t[i] / fil_length) * 0.8))
            r_width = max(1, int(fil_width * (1 - t[i] / fil_length * 0.7)))

            for ddy in range(-r_width, r_width + 1):
                for ddx in range(-r_width, r_width + 1):
                    ny, nx = py + ddy, px + ddx
                    if 0 <= ny < h and 0 <= nx < w:
                        if rng.random() < 0.7:
                            tensor[ny, nx] = max(tensor[ny, nx], brightness)

    scatter_mask = in_lobe & (r < length) & (r >= nucleus_radius * 0.5)
    scatter_prob = np.where(scatter_mask, 0.02 * (1 - r / length), 0.0)
    tensor[scatter_mask & (noise < scatter_prob)] = np.where(
        tensor[scatter_mask & (noise < scatter_prob)] < 200,
        rng.integers(150, 255, size=np.count_nonzero(scatter_mask & (noise < scatter_prob))).astype(np.uint8),
        tensor[scatter_mask & (noise < scatter_prob)],
    )


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    tensor = create_canvas(height, width)
    center = random_center(height, width)

    base_length = min(height, width) * rng.uniform(0.25, 0.45)

    cut_angle = rng.uniform(-10, 10)
    cut_length = base_length * rng.uniform(0.8, 1.2)
    cut_spread = rng.uniform(15, 30)

    draw_lobe(tensor, center, cut_angle, cut_length, cut_spread, rng)
    draw_lobe(tensor, center, cut_angle + 180, cut_length * rng.uniform(0.7, 1.0), cut_spread, rng)

    explosion_angle = cut_angle + 90 + rng.uniform(-10, 10)
    explosion_length = base_length * rng.uniform(1.0, 1.5)
    explosion_spread = rng.uniform(20, 40)

    draw_lobe(tensor, center, explosion_angle, explosion_length, explosion_spread, rng)

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
        print(f"  Centro y dimensiones del tensor: {tensor.shape}")


if __name__ == "__main__":
    main()
