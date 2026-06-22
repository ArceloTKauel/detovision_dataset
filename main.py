import numpy as np
from PIL import Image

HEIGHT = 720
WIDTH = 1280


def perlin_noise_2d(shape: tuple[int, int], scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    h, w = shape
    noise = np.zeros((h, w))

    for octave in range(octaves):
        freq = 2 ** octave
        amp = 0.5 ** octave
        grid_h = max(2, int(h / (scale / freq)))
        grid_w = max(2, int(w / (scale / freq)))

        angles = rng.uniform(0, 2 * np.pi, (grid_h + 1, grid_w + 1))
        gradients_y = np.sin(angles)
        gradients_x = np.cos(angles)

        ys = np.linspace(0, grid_h - 1, h, endpoint=False)
        xs = np.linspace(0, grid_w - 1, w, endpoint=False)

        y0 = ys.astype(int)
        x0 = xs.astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        fy = ys - y0
        fx = xs - x0

        fy_smooth = fy * fy * (3 - 2 * fy)
        fx_smooth = fx * fx * (3 - 2 * fx)

        fy_s = fy_smooth[:, np.newaxis]
        fx_s = fx_smooth[np.newaxis, :]
        fy_col = fy[:, np.newaxis]
        fx_row = fx[np.newaxis, :]

        y0_col = y0[:, np.newaxis]
        x0_row = x0[np.newaxis, :]
        y1_col = y1[:, np.newaxis]
        x1_row = x1[np.newaxis, :]

        d00_y = fy_col
        d00_x = fx_row
        d10_y = fy_col - 1
        d10_x = fx_row
        d01_y = fy_col
        d01_x = fx_row - 1
        d11_y = fy_col - 1
        d11_x = fx_row - 1

        g00 = gradients_y[y0_col, x0_row] * d00_y + gradients_x[y0_col, x0_row] * d00_x
        g10 = gradients_y[y1_col, x0_row] * d10_y + gradients_x[y1_col, x0_row] * d10_x
        g01 = gradients_y[y0_col, x1_row] * d01_y + gradients_x[y0_col, x1_row] * d01_x
        g11 = gradients_y[y1_col, x1_row] * d11_y + gradients_x[y1_col, x1_row] * d11_x

        lerp_x0 = g00 + fx_s * (g01 - g00)
        lerp_x1 = g10 + fx_s * (g11 - g10)
        value = lerp_x0 + fy_s * (lerp_x1 - lerp_x0)

        noise += value * amp

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


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


def distribute_centers_in_quadrilateral(
    vertices: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    v0, v1, v2, v3 = vertices
    centers = []
    for _ in range(num_points):
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
        centers.append((int(round(point[0])), int(round(point[1]))))
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

    fringe_radius = smoke_radius * 1.3
    min_y = max(0, int(all_cy.min() - fringe_radius - 10))
    max_y = min(h, int(all_cy.max() + fringe_radius + 10))
    min_x = max(0, int(all_cx.min() - fringe_radius - 10))
    max_x = min(w, int(all_cx.max() + fringe_radius + 10))

    region_h = max_y - min_y
    region_w = max_x - min_x
    if region_h <= 0 or region_w <= 0:
        return

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    min_dist = np.full(ys.shape, np.inf)
    for cy, cx in centers:
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        min_dist = np.minimum(min_dist, dist)

    region = tensor[min_y:max_y, min_x:max_x]

    perlin = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.4, rng=rng, octaves=5)
    perlin_fine = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.15, rng=rng, octaves=3)

    radius_distortion = 1.0 + (perlin - 0.5) * 0.6
    distorted_dist = min_dist / radius_distortion

    core_radius = smoke_radius * 0.25
    core_mask = distorted_dist < core_radius
    core_brightness = (255 * (0.85 + perlin_fine[core_mask] * 0.15)).clip(0, 255).astype(np.uint8)
    region[core_mask] = np.maximum(region[core_mask], core_brightness)

    mid_radius = smoke_radius * 0.5
    mid_mask = (distorted_dist >= core_radius) & (distorted_dist < mid_radius)
    ratio_mid = (distorted_dist[mid_mask] - core_radius) / (mid_radius - core_radius)
    prob_mid = 0.9 - 0.4 * ratio_mid
    brightness_mid = (255 * (1 - ratio_mid * 0.3) * (0.7 + perlin[mid_mask] * 0.3)).clip(0, 255).astype(np.uint8)
    region[mid_mask] = np.where(
        perlin[mid_mask] > (1 - prob_mid),
        np.maximum(region[mid_mask], brightness_mid),
        region[mid_mask],
    )

    outer_mask = (distorted_dist >= mid_radius) & (distorted_dist < smoke_radius)
    ratio_outer = (distorted_dist[outer_mask] - mid_radius) / (smoke_radius - mid_radius)
    prob_outer = 0.6 * (1 - ratio_outer) ** 2
    brightness_outer = (200 * (1 - ratio_outer * 0.7) * (0.6 + perlin[outer_mask] * 0.4)).clip(0, 255).astype(np.uint8)
    region[outer_mask] = np.where(
        perlin[outer_mask] > (1 - prob_outer),
        np.maximum(region[outer_mask], brightness_outer),
        region[outer_mask],
    )

    fringe_mask = (distorted_dist >= smoke_radius) & (distorted_dist < fringe_radius)
    ratio_fringe = (distorted_dist[fringe_mask] - smoke_radius) / (fringe_radius - smoke_radius)
    prob_fringe = 0.15 * (1 - ratio_fringe) ** 3
    brightness_fringe = (120 * (1 - ratio_fringe) * perlin[fringe_mask]).clip(0, 255).astype(np.uint8)
    region[fringe_mask] = np.where(
        perlin[fringe_mask] > (1 - prob_fringe),
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

    num_centers = rng.integers(40, 80)
    centers = distribute_centers_in_quadrilateral(quad, num_centers, rng)

    for center in centers:
        draw_center(tensor, center, rng.integers(0, 2))

    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    smoke_radius = base_length * rng.uniform(0.15, 0.3)
    draw_smoke(tensor, centers, smoke_radius, rng)

    tensor = np.where(tensor >= 128, 255, 0).astype(np.uint8)

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
