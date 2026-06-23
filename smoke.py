import numpy as np
from PIL import Image, ImageDraw

from perlin_noise import perlin_noise_2d


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

    subtractive = perlin_noise_2d(
        (region_h, region_w), scale=smoke_radius * 0.3, rng=rng, octaves=3
    )
    smoke_mask = region > 0
    candidate_mask = (subtractive < 0.45) & smoke_mask
    candidate_mask &= distorted_dist > core_radius * 0.5

    candidate_coords = np.argwhere(candidate_mask)
    if len(candidate_coords) > 0:
        num_polys = min(len(candidate_coords), rng.integers(8, 25))
        seed_indices = rng.choice(len(candidate_coords), size=num_polys, replace=False)
        seeds = candidate_coords[seed_indices]

        poly_img = Image.new("L", (region_w, region_h), 0)
        draw = ImageDraw.Draw(poly_img)

        for seed in seeds:
            sy, sx = seed
            dist = distorted_dist[sy, sx]
            core_factor = float(np.clip(dist / (core_radius * 2), 0.2, 1.0))

            num_verts = rng.integers(4, 8)
            angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_verts))
            base_r = smoke_radius * rng.uniform(0.1, 0.3) * core_factor

            verts = []
            for a in angles:
                r = base_r * rng.uniform(0.6, 1.0)
                vy = sy + r * np.sin(a)
                vx = sx + r * np.cos(a)
                verts.append((int(round(vx)), int(round(vy))))

            draw.polygon(verts, fill=255)

        poly_array = np.array(poly_img)
        region[poly_array > 0] = 0


def measure_smoke_width(
    tensor: np.ndarray,
    origin: tuple[int, int],
    angle: float,
) -> float:
    h, w = tensor.shape
    oy, ox = origin
    dist = 0.0
    step = 1.0

    while True:
        dist += step
        py = int(round(oy + dist * np.sin(angle)))
        px = int(round(ox + dist * np.cos(angle)))

        if py < 0 or py >= h or px < 0 or px >= w:
            break
        if tensor[py, px] == 0:
            break

    return dist
