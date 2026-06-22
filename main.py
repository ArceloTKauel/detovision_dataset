import numpy as np
from PIL import Image

HEIGHT = 720
WIDTH = 1280


def create_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def random_center(height: int, width: int, rng: np.random.Generator, margin: float = 0.15) -> tuple[int, int]:
    cy = rng.integers(int(height * margin), int(height * (1 - margin)))
    cx = rng.integers(int(width * margin), int(width * (1 - margin)))
    return int(cy), int(cx)


def perturbed_radius(base_radius: float, theta: np.ndarray, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    result = np.ones_like(theta) * base_radius
    for i in range(octaves):
        freq = 2 ** i
        amp = base_radius * 0.3 / (i + 1)
        phase = rng.uniform(0, 2 * np.pi)
        result += amp * np.sin(freq * theta + phase)
    return result


def draw_nucleus(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle_rad: float,
    spread_rad: float,
    base_radius: float,
    rng: np.random.Generator,
    is_explosion: bool = False,
) -> None:
    h, w = tensor.shape
    cy, cx = center

    ys, xs = np.mgrid[0:h, 0:w]
    dy = (ys - cy).astype(np.float64)
    dx = (xs - cx).astype(np.float64)
    r = np.sqrt(dy**2 + dx**2)
    theta = np.arctan2(dy, dx)

    angle_diff = np.abs(np.arctan2(np.sin(theta - angle_rad), np.cos(theta - angle_rad)))

    noise_boundary = perturbed_radius(spread_rad, theta, rng, octaves=4)
    in_lobe = angle_diff < noise_boundary

    p_radius = perturbed_radius(base_radius, theta, rng, octaves=6)

    core_r = p_radius * 0.5
    core_mask = in_lobe & (r < core_r)
    noise = rng.random(tensor.shape)
    tensor[core_mask & (noise > 0.05)] = 255

    holes = rng.random(tensor.shape)
    tensor[core_mask & (holes < 0.08)] = 0

    mid_mask = in_lobe & (r >= core_r) & (r < p_radius * 0.8)
    mid_prob = np.where(mid_mask, 0.85 - 0.4 * (r / p_radius), 0.0)
    tensor[mid_mask & (noise < mid_prob)] = 255

    outer_mask = in_lobe & (r >= p_radius * 0.8) & (r < p_radius)
    outer_prob = np.where(outer_mask, 0.5 * (1 - r / p_radius) ** 2, 0.0)
    tensor[outer_mask & (noise < outer_prob)] = 255

    if is_explosion:
        extra_spread = spread_rad * 1.3
        extra_in = angle_diff < extra_spread
        extra_mask = extra_in & (r >= p_radius * 0.3) & (r < p_radius * 1.1)
        extra_prob = np.where(extra_mask, 0.15 * (1 - r / (p_radius * 1.1)), 0.0)
        tensor[extra_mask & (noise < extra_prob)] = rng.integers(
            180, 255, size=np.count_nonzero(extra_mask & (noise < extra_prob))
        ).astype(np.uint8)


def draw_filaments(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle_rad: float,
    spread_rad: float,
    max_length: float,
    rng: np.random.Generator,
    num_filaments: int | None = None,
) -> None:
    h, w = tensor.shape
    cy, cx = center

    if num_filaments is None:
        num_filaments = rng.integers(25, 55)

    for _ in range(num_filaments):
        fil_angle = angle_rad + rng.uniform(-spread_rad * 0.95, spread_rad * 0.95)
        fil_length = max_length * rng.uniform(0.4, 1.0)
        fil_width_base = rng.uniform(1.0, 3.0)
        curvature = rng.uniform(-0.003, 0.003)

        num_points = int(fil_length * 1.5)
        if num_points < 2:
            continue

        t = np.linspace(0, fil_length, num_points)
        current_angle = fil_angle
        wobble_freq = rng.uniform(0.01, 0.05)
        wobble_amp = rng.uniform(1.0, 4.0)

        positions_x = np.zeros(num_points)
        positions_y = np.zeros(num_points)
        positions_x[0] = cx
        positions_y[0] = cy

        for i in range(1, num_points):
            dt = t[i] - t[i - 1]
            current_angle += curvature * dt
            wobble = wobble_amp * np.sin(wobble_freq * t[i])
            positions_x[i] = positions_x[i - 1] + dt * np.cos(current_angle) + wobble * np.cos(current_angle + np.pi / 2) * 0.3
            positions_y[i] = positions_y[i - 1] + dt * np.sin(current_angle) + wobble * np.sin(current_angle + np.pi / 2) * 0.3

        for i in range(num_points):
            progress = t[i] / fil_length
            px = int(round(positions_x[i]))
            py = int(round(positions_y[i]))
            brightness = int(255 * (1 - progress * 0.85) ** 1.5)
            brightness = max(40, brightness)
            r_width = max(0, int(fil_width_base * (1 - progress * 0.8)))

            if rng.random() < (1 - progress * 0.6):
                for ddy in range(-r_width, r_width + 1):
                    for ddx in range(-r_width, r_width + 1):
                        ny, nx = py + ddy, px + ddx
                        if 0 <= ny < h and 0 <= nx < w:
                            if rng.random() < 0.8:
                                tensor[ny, nx] = max(tensor[ny, nx], brightness)


def draw_shockwave_arcs(
    tensor: np.ndarray,
    center: tuple[int, int],
    base_radius: float,
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center

    num_arcs = rng.integers(2, 5)
    for _ in range(num_arcs):
        arc_radius = base_radius * rng.uniform(0.8, 2.5)
        arc_start = rng.uniform(0, 2 * np.pi)
        arc_span = rng.uniform(np.pi * 0.3, np.pi * 1.2)
        thickness = rng.uniform(0.5, 1.5)

        num_points = int(arc_span * arc_radius * 0.5)
        if num_points < 2:
            continue

        angles = np.linspace(arc_start, arc_start + arc_span, num_points)
        for angle in angles:
            r_offset = rng.normal(0, 1.0)
            px = int(round(cx + (arc_radius + r_offset) * np.cos(angle)))
            py = int(round(cy + (arc_radius + r_offset) * np.sin(angle)))

            if 0 <= py < h and 0 <= px < w:
                if rng.random() < 0.6:
                    brightness = rng.integers(30, 100)
                    for ddy in range(int(-thickness), int(thickness) + 1):
                        for ddx in range(int(-thickness), int(thickness) + 1):
                            ny, nx = py + ddy, px + ddx
                            if 0 <= ny < h and 0 <= nx < w:
                                tensor[ny, nx] = max(tensor[ny, nx], brightness)


def draw_debris(
    tensor: np.ndarray,
    center: tuple[int, int],
    max_radius: float,
    rng: np.random.Generator,
) -> None:
    h, w = tensor.shape
    cy, cx = center

    num_particles = rng.integers(200, 600)
    for _ in range(num_particles):
        angle = rng.uniform(0, 2 * np.pi)
        dist = max_radius * rng.uniform(0.2, 2.0)
        px = int(round(cx + dist * np.cos(angle)))
        py = int(round(cy + dist * np.sin(angle)))

        if 0 <= py < h and 0 <= px < w:
            brightness = rng.integers(40, 200)
            size = rng.integers(0, 2)
            for ddy in range(-size, size + 1):
                for ddx in range(-size, size + 1):
                    ny, nx = py + ddy, px + ddx
                    if 0 <= ny < h and 0 <= nx < w:
                        tensor[ny, nx] = max(tensor[ny, nx], brightness)


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    tensor = create_canvas(height, width)
    center = random_center(height, width, rng)

    base_length = min(height, width) * rng.uniform(0.25, 0.40)

    cut_angle_deg = rng.uniform(-10, 10)
    cut_angle = np.radians(cut_angle_deg)
    cut_length = base_length * rng.uniform(0.7, 1.0)
    cut_spread = np.radians(rng.uniform(15, 25))
    nucleus_radius = cut_length * 0.4

    draw_nucleus(tensor, center, cut_angle, cut_spread, nucleus_radius, rng)
    draw_nucleus(tensor, center, cut_angle + np.pi, cut_spread * rng.uniform(0.7, 1.0), nucleus_radius * rng.uniform(0.6, 0.9), rng)

    draw_filaments(tensor, center, cut_angle, cut_spread * 1.1, cut_length, rng)
    draw_filaments(tensor, center, cut_angle + np.pi, cut_spread * 1.1, cut_length * rng.uniform(0.6, 0.9), rng)

    explosion_angle_deg = cut_angle_deg + 90 + rng.uniform(-10, 10)
    explosion_angle = np.radians(explosion_angle_deg)
    explosion_length = base_length * rng.uniform(1.0, 1.5)
    explosion_spread = np.radians(rng.uniform(25, 45))
    explosion_nucleus_radius = explosion_length * 0.35

    draw_nucleus(tensor, center, explosion_angle, explosion_spread, explosion_nucleus_radius, rng, is_explosion=True)
    draw_filaments(tensor, center, explosion_angle, explosion_spread * 1.2, explosion_length, rng, num_filaments=rng.integers(30, 60))

    draw_shockwave_arcs(tensor, center, base_length * 0.6, rng)
    draw_debris(tensor, center, base_length, rng)

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
