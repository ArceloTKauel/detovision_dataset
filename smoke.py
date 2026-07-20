"""
smoke.py - Generación de humo con textura orgánica y manchas sustractivas.

Dibuja el humo de la explosión alrededor de los centros de impacto usando
un modelo de zonas concéntricas (core, mid, outer, fringe) con distorsión
Perlin para bordes irregulares. Luego genera polígonos aleatorios que
"recortan" huecos dentro del humo, simulando manchas/huecos realistas.

Funciones:
    - draw_smoke(tensor, centers, smoke_radius, rng, mask): Dibuja el humo
      completo sobre el tensor. Usa distancia mínima a cualquier centro para
      determinar la zona, y Perlin noise para distorsionar radios y variar
      brillo. Si se pasa mask, marca los píxeles de humo como clase 1
      (y borra de la mask donde caen las manchas sustractivas).
    - measure_smoke_width(tensor, origin, angle): Mide la distancia desde el
      origen en una dirección dada hasta encontrar un píxel negro. Usado por
      trajectories.py para que las trayectorias empiecen fuera del humo.
"""

import numpy as np
from PIL import Image, ImageDraw

from perlin_noise import perlin_noise_2d

# Escala de brillo global de la explosión: sortea una vez por imagen (en
# main.py, vía sample_brightness_scale) y se aplica tanto al humo como a los
# centros camuflados debajo (ver canvas.py::draw_center), para que ninguno
# sature a blanco pleno. Acerca el rango dinámico al de las referencias sin
# binarizar (mascara_cambios_final_sinbin_*), que nunca llegan a blanco puro.
SMOKE_BRIGHTNESS_SCALE_RANGE = (0.45, 0.70)

# Rango objetivo de la clase humo (medido con pixel_inspector_gui.py): 15 a
# 255. _SMOKE_BRIGHTNESS_FLOOR evita que cualquier píxel dibujado como humo
# quede por debajo de 15 (se aplica en el clip de cada zona). El resto del
# pipeline (brightness_scale, grain, etc.) ya mantiene el grueso del humo
# bien por debajo de 255; los "flecos" definidos más abajo son la única vía
# para que un píxel puntual llegue cerca de blanco pleno, y con probabilidad
# baja a propósito (ver _HOT_FLECK_PROB).
_SMOKE_BRIGHTNESS_FLOOR = 15

# Flecos brillantes: chispas/reflejos puntuales muy poco frecuentes que
# rompen por encima del techo habitual del humo (~180) hasta casi blanco
# pleno, independientes de brightness_scale (si no, nunca podrían acercarse
# a 255 en una imagen con brightness_scale bajo). La probabilidad es baja
# para que la gran mayoría del humo se mantenga por debajo de 200.
_HOT_FLECK_PROB = 0.01
_HOT_FLECK_RANGE = (200, 255)


def sample_brightness_scale(rng: np.random.Generator) -> float:
    """Sortea la escala de brillo global de una explosión (una vez por imagen)."""
    return rng.uniform(*SMOKE_BRIGHTNESS_SCALE_RANGE)


def draw_smoke(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
) -> None:
    h, w = tensor.shape

    all_cy = np.array([c[0] for c in centers], dtype=np.float64)
    all_cx = np.array([c[1] for c in centers], dtype=np.float64)

    # Región de trabajo: bounding box de todos los centros + margen del radio
    fringe_radius = smoke_radius * 1.3
    min_y = max(0, int(all_cy.min() - fringe_radius - 10))
    max_y = min(h, int(all_cy.max() + fringe_radius + 10))
    min_x = max(0, int(all_cx.min() - fringe_radius - 10))
    max_x = min(w, int(all_cx.max() + fringe_radius + 10))

    region_h = max_y - min_y
    region_w = max_x - min_x
    if region_h <= 0 or region_w <= 0:
        return

    # Grilla de coordenadas de la región
    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    # Para cada píxel, calcular la distancia mínima a cualquier centro
    min_dist = np.full(ys.shape, np.inf)
    for cy, cx in centers:
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        min_dist = np.minimum(min_dist, dist)

    region = tensor[min_y:max_y, min_x:max_x]
    mask_region = mask[min_y:max_y, min_x:max_x] if mask is not None else None

    # Perlin noise para distorsionar los bordes del humo y variar el brillo
    perlin = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.4, rng=rng, octaves=5)
    perlin_fine = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.15, rng=rng, octaves=3)

    # Grano fino: escala mucho más chica que perlin_fine, para romper la
    # superficie lisa incluso dentro del core (textura fibrosa/granulada en
    # vez de un relleno plano). Se multiplica sobre el brillo de TODAS las
    # zonas junto con brightness_scale.
    grain = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.05, rng=rng, octaves=2)
    grain_factor = 0.55 + 0.45 * grain

    # Distorsión del radio: el Perlin desplaza el borde ±30%, creando irregularidad
    radius_distortion = 1.0 + (perlin - 0.5) * 0.6
    distorted_dist = min_dist / radius_distortion

    # === ZONA CORE (0 a 25% del radio) ===
    # Zona más brillante y densa, casi sólida
    core_radius = smoke_radius * 0.25
    # Banda de transición centrada en core_radius: sin ella, el brillo (perlin_fine
    # vs perlin) y la probabilidad de dibujo (100% vs ~90%) cambian de golpe justo
    # en el borde del core, lo que se ve como una costura/corte. Dentro de la banda
    # se mezclan ambos ruidos y se mantiene dibujo 100% (sin agujeros) para que la
    # transición sea gradual.
    seam_half_width = smoke_radius * 0.06
    seam_lo = core_radius - seam_half_width
    seam_hi = core_radius + seam_half_width

    core_mask = distorted_dist < seam_lo
    core_brightness = (
        255 * (0.85 + perlin_fine[core_mask] * 0.15) * brightness_scale * grain_factor[core_mask]
    ).clip(_SMOKE_BRIGHTNESS_FLOOR, 255).astype(np.uint8)
    region[core_mask] = np.maximum(region[core_mask], core_brightness)
    if mask_region is not None:
        mask_region[core_mask] = 1

    # === ZONA DE TRANSICIÓN (seam_lo a seam_hi) ===
    seam_mask = (distorted_dist >= seam_lo) & (distorted_dist < seam_hi)
    t = (distorted_dist[seam_mask] - seam_lo) / (seam_hi - seam_lo)
    t = t * t * (3 - 2 * t)  # smoothstep: 0 en seam_lo, 1 en seam_hi
    perlin_seam = perlin_fine[seam_mask] * (1 - t) + perlin[seam_mask] * t
    seam_brightness = (
        255 * (1 - 0.3 * t) * (0.85 + perlin_seam * 0.15) * brightness_scale * grain_factor[seam_mask]
    ).clip(_SMOKE_BRIGHTNESS_FLOOR, 255).astype(np.uint8)
    region[seam_mask] = np.maximum(region[seam_mask], seam_brightness)
    if mask_region is not None:
        mask_region[seam_mask] = 1

    # === ZONA MID (seam_hi a 50% del radio) ===
    # Densidad decreciente, probabilidad de dibujar baja con la distancia
    mid_radius = smoke_radius * 0.5
    mid_mask = (distorted_dist >= seam_hi) & (distorted_dist < mid_radius)
    ratio_mid = (distorted_dist[mid_mask] - seam_hi) / (mid_radius - seam_hi)
    prob_mid = 0.9 - 0.4 * ratio_mid  # probabilidad de 90% a 50%
    brightness_mid = (
        255 * (1 - ratio_mid * 0.3) * (0.7 + perlin[mid_mask] * 0.3) * brightness_scale * grain_factor[mid_mask]
    ).clip(_SMOKE_BRIGHTNESS_FLOOR, 255).astype(np.uint8)
    mid_drawn = perlin[mid_mask] > (1 - prob_mid)
    region[mid_mask] = np.where(
        mid_drawn,
        np.maximum(region[mid_mask], brightness_mid),
        region[mid_mask],
    )
    if mask_region is not None:
        mid_indices = np.argwhere(mid_mask)
        mask_region[mid_indices[mid_drawn, 0], mid_indices[mid_drawn, 1]] = 1

    # === ZONA OUTER (50% a 100% del radio) ===
    # Humo disperso, probabilidad decae cuadráticamente
    outer_mask = (distorted_dist >= mid_radius) & (distorted_dist < smoke_radius)
    ratio_outer = (distorted_dist[outer_mask] - mid_radius) / (smoke_radius - mid_radius)
    prob_outer = 0.6 * (1 - ratio_outer) ** 2
    brightness_outer = (
        200 * (1 - ratio_outer * 0.7) * (0.6 + perlin[outer_mask] * 0.4) * brightness_scale * grain_factor[outer_mask]
    ).clip(_SMOKE_BRIGHTNESS_FLOOR, 255).astype(np.uint8)
    outer_drawn = perlin[outer_mask] > (1 - prob_outer)
    region[outer_mask] = np.where(
        outer_drawn,
        np.maximum(region[outer_mask], brightness_outer),
        region[outer_mask],
    )
    if mask_region is not None:
        outer_indices = np.argwhere(outer_mask)
        mask_region[outer_indices[outer_drawn, 0], outer_indices[outer_drawn, 1]] = 1

    # === ZONA FRINGE (100% a 130% del radio) ===
    # Borde difuso, muy baja probabilidad, partículas sueltas
    fringe_mask = (distorted_dist >= smoke_radius) & (distorted_dist < fringe_radius)
    ratio_fringe = (distorted_dist[fringe_mask] - smoke_radius) / (fringe_radius - smoke_radius)
    prob_fringe = 0.15 * (1 - ratio_fringe) ** 3
    brightness_fringe = (
        120 * (1 - ratio_fringe) * perlin[fringe_mask] * brightness_scale * grain_factor[fringe_mask]
    ).clip(_SMOKE_BRIGHTNESS_FLOOR, 255).astype(np.uint8)
    fringe_drawn = perlin[fringe_mask] > (1 - prob_fringe)
    region[fringe_mask] = np.where(
        fringe_drawn,
        np.maximum(region[fringe_mask], brightness_fringe),
        region[fringe_mask],
    )
    if mask_region is not None:
        fringe_indices = np.argwhere(fringe_mask)
        mask_region[fringe_indices[fringe_drawn, 0], fringe_indices[fringe_drawn, 1]] = 1

    # === FLECOS BRILLANTES (chispas raras cercanas a blanco pleno) ===
    # Aplicado sobre el humo ya dibujado (todas las zonas), independiente de
    # brightness_scale a propósito: es la única vía para que un píxel de
    # humo llegue cerca de 255 aunque esta explosión haya salido "apagada".
    smoke_so_far = region > 0
    fleck_roll = rng.random(region.shape)
    fleck_mask = smoke_so_far & (fleck_roll < _HOT_FLECK_PROB)
    if fleck_mask.any():
        fleck_brightness = rng.uniform(*_HOT_FLECK_RANGE, size=int(fleck_mask.sum())).astype(np.uint8)
        region[fleck_mask] = np.maximum(region[fleck_mask], fleck_brightness)

    # === MANCHAS SUSTRACTIVAS (polígonos que recortan huecos en el humo) ===
    # Se generan en zonas donde el Perlin sustractivo es bajo (< 0.45),
    # evitando el core central para no romper la forma base.
    subtractive = perlin_noise_2d(
        (region_h, region_w), scale=smoke_radius * 0.3, rng=rng, octaves=3
    )
    smoke_mask = region > 0
    candidate_mask = (subtractive < 0.45) & smoke_mask
    candidate_mask &= distorted_dist > core_radius * 0.5  # proteger el core

    candidate_coords = np.argwhere(candidate_mask)
    if len(candidate_coords) > 0:
        num_polys = min(len(candidate_coords), rng.integers(15, 45))
        seed_indices = rng.choice(len(candidate_coords), size=num_polys, replace=False)
        seeds = candidate_coords[seed_indices]

        # Dibujar polígonos en una imagen auxiliar para luego aplicar como máscara
        poly_img = Image.new("L", (region_w, region_h), 0)
        draw = ImageDraw.Draw(poly_img)

        for seed in seeds:
            sy, sx = seed
            dist = distorted_dist[sy, sx]
            # Polígonos más pequeños cerca del core, más grandes lejos
            core_factor = float(np.clip(dist / (core_radius * 2), 0.2, 1.0))

            # Polígono irregular: vértices a ángulos ordenados con radio variable
            num_verts = rng.integers(4, 14)
            angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_verts))
            base_r = smoke_radius * rng.uniform(0.1, 0.3) * core_factor

            verts = []
            for a in angles:
                r = base_r * rng.uniform(0.6, 1.0)  # variación del radio por vértice
                vy = sy + r * np.sin(a)
                vx = sx + r * np.cos(a)
                verts.append((int(round(vx)), int(round(vy))))

            draw.polygon(verts, fill=255)

        # Aplicar máscara: donde hay polígono, borrar el humo
        poly_array = np.array(poly_img)
        region[poly_array > 0] = 0
        if mask_region is not None:
            mask_region[poly_array > 0] = 0


def measure_smoke_width(
    tensor: np.ndarray,
    origin: tuple[int, int],
    angle: float,
) -> float:
    """
    Mide cuántos píxeles de humo hay desde el origen en la dirección dada.
    Avanza píxel a píxel hasta encontrar fondo negro o salir del lienzo.
    Usado para que las trayectorias empiecen donde termina el humo.
    """
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
