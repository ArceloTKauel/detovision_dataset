"""
terrain.py - Manchón de parallax de terreno (ego-motion de cámara).

Simula la textura de fondo que aparece en video real por el movimiento de la
cámara sobre el relieve del terreno al diferenciar frames consecutivos (ver
inference/inputs/mascara_cambios_final_sinbin_*.png en detovision_segmentation,
donde el modelo la confundía con la clase trayectoria). Es un gradiente
CONTINUO de todo el cuadro, no una forma discreta como el humo o los centros.

Se dibuja PRIMERO en el pipeline (antes de draw_center/draw_smoke/
draw_trajectories, ver main.py::generate_explosion) para que el resto de los
elementos lo ocluyan de forma natural vía la composición np.maximum ya
existente en canvas.py/smoke.py, sin necesitar ninguna lógica de exclusión
propia. Por eso draw_terrain no recibe (ni toca) `mask`: donde humo/
trayectoria/derrumbe efectivamente cubren un píxel, sus propias funciones ya
marcan esa clase en la máscara sin mirar el tensor de fondo (ver
smoke.py::draw_smoke, que marca mask_region según su propia geometría,
independiente del valor previo del tensor) — el terreno solo "gana" la
etiqueta de fondo en las zonas que ningún otro elemento cubre, que es
exactamente lo deseado (es un artefacto, no un objeto real a segmentar).

Prototipo y validación visual original contra las referencias reales en
detovision_segmentation/scripts/preview_terrain_lines.py y
detovision_segmentation/utils/dataset.py (familia FAKE_TERRAIN_*, aplicada ahí
como augmentation "en caliente" en cada época de entrenamiento). Esta versión
es la misma técnica portada al patrón `rng: np.random.Generator` de este
repo, para hornearla en la generación base del dataset (fija por imagen, no
resorteada por época) — ver decisión de proyecto: ambas capas conviven
(la de acá + la "en caliente" del otro repo, que sigue activa sin cambios).

Funciones:
    - draw_terrain(tensor, rng, prob): Sortea si esta imagen lleva manchón de
      terreno y, si corresponde, lo dibuja sobre el tensor.
"""

import numpy as np
from PIL import Image, ImageFilter

# Probabilidad de que una imagen del dataset lleve manchón de terreno —
# mismo valor que FAKE_TERRAIN_PROB en detovision_segmentation/utils/dataset.py,
# pero acá la decisión queda fija para siempre en esa imagen (no se
# resortea por época).
TERRAIN_PROB = 0.5

# ── Campo de "elevación" (dirección/curvatura del parallax) ────────────────
TERRAIN_FIELD_OCTAVES    = (1, 2)
TERRAIN_FIELD_CELL_RANGE = (250.0, 500.0)
TERRAIN_PERTURB_STRENGTH = (0.01, 0.05)
TERRAIN_VANISHING_MARGIN = 1.5

# ── Manchón: el campo se usa directo como brillo, modulado en bandas suaves ─
TERRAIN_BAND_PERIOD   = (0.04, 0.12)
TERRAIN_BLOTCH_FLOOR  = (0.35, 0.60)
TERRAIN_BLOTCH_MAX_VAL = (15, 55)
TERRAIN_BLUR_RADIUS   = (0.3, 0.8)

# Grano fibroso de alta frecuencia, multiplicado sobre la banda.
TERRAIN_GRAIN_OCTAVES    = (3, 5)
TERRAIN_GRAIN_CELL_RANGE = (4.0, 12.0)
TERRAIN_GRAIN_CONTRAST   = (0.5, 0.9)

# Intensidad global por muestra.
TERRAIN_INTENSITY_RANGE = (0.05, 1.0)


def _multi_octave_field(h, w, octaves_range, cell_range, rng):
    """Campo [h, w] de ruido fractal multi-octava — misma técnica que
    _multi_octave_field en detovision_segmentation/utils/dataset.py, portada a
    `rng.random` en vez de `np.random.rand` (estado global) para que la
    generación quede determinada por el seed por índice de generate_dataset.py."""
    octaves = rng.integers(*octaves_range, endpoint=True)
    base_cell = rng.uniform(*cell_range)

    field = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amplitude = 1.0, 0.0
    for octave in range(octaves):
        cell = max(2.0, base_cell / (2 ** octave))
        grid_h = max(2, round(h / cell))
        grid_w = max(2, round(w / cell))
        small = rng.random((grid_h, grid_w)).astype(np.float32)
        layer = np.array(Image.fromarray(small).resize((w, h), Image.BILINEAR))
        field += layer * amplitude
        total_amplitude += amplitude
        amplitude *= 0.5

    field /= total_amplitude
    return field


def _stretch_to_unit_range(field):
    field = field - field.min()
    return field / (field.max() + 1e-8)


def _terrain_elevation_field(h, w, rng):
    """Campo de "elevación" = distancia a un punto de fuga sorteado + perturbación
    fractal (rompe la simetría perfecta del campo radial)."""
    vy = rng.uniform(-TERRAIN_VANISHING_MARGIN * h, (1 + TERRAIN_VANISHING_MARGIN) * h)
    vx = rng.uniform(-TERRAIN_VANISHING_MARGIN * w, (1 + TERRAIN_VANISHING_MARGIN) * w)

    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt((yy - vy) ** 2 + (xx - vx) ** 2).astype(np.float32)
    radial /= radial.max()

    perturbation = _stretch_to_unit_range(
        _multi_octave_field(h, w, TERRAIN_FIELD_OCTAVES, TERRAIN_FIELD_CELL_RANGE, rng))
    strength = rng.uniform(*TERRAIN_PERTURB_STRENGTH)

    return radial + strength * (perturbation - 0.5)


def _terrain_grain_multiplier(h, w, rng):
    """Multiplicador de textura fibrosa de alta frecuencia, centrado en 1.0 — ver
    _fake_terrain_grain_multiplier en detovision_segmentation/utils/dataset.py."""
    field = _multi_octave_field(h, w, TERRAIN_GRAIN_OCTAVES, TERRAIN_GRAIN_CELL_RANGE, rng)
    contrast = rng.uniform(*TERRAIN_GRAIN_CONTRAST)
    return 1.0 - contrast + 2.0 * contrast * field


def _terrain_blotch_brightness(h, w, rng):
    """Mapa de brillo [0, 255] con bandas suaves (modulación seno) + piso +
    grano fibroso + intensidad global — ver _fake_terrain_blotch_brightness en
    detovision_segmentation/utils/dataset.py para el razonamiento completo."""
    field = _terrain_elevation_field(h, w, rng)
    lo, hi = field.min(), field.max()
    normalized = (field - lo) / (hi - lo + 1e-8)

    period = rng.uniform(*TERRAIN_BAND_PERIOD)
    phase = rng.uniform(0, 2 * np.pi)
    banding = 0.5 + 0.5 * np.sin(2 * np.pi * normalized / period + phase)

    floor = rng.uniform(*TERRAIN_BLOTCH_FLOOR)
    banding = np.clip((banding - floor) / (1.0 - floor), 0.0, 1.0)

    banding = np.clip(banding * _terrain_grain_multiplier(h, w, rng), 0.0, 1.0)

    intensity = rng.uniform(*TERRAIN_INTENSITY_RANGE)
    max_val = rng.uniform(*TERRAIN_BLOTCH_MAX_VAL) * intensity
    brightness_img = Image.fromarray((banding * max_val).astype(np.uint8))

    blur_radius = rng.uniform(*TERRAIN_BLUR_RADIUS)
    brightness_img = brightness_img.filter(ImageFilter.GaussianBlur(blur_radius))

    return np.array(brightness_img, dtype=np.float32)


def draw_terrain(
    tensor: np.ndarray,
    rng: np.random.Generator,
    prob: float = TERRAIN_PROB,
) -> None:
    """Sortea si esta imagen lleva manchón de terreno (probabilidad `prob`) y,
    si corresponde, lo dibuja sobre `tensor` vía np.maximum (mismo criterio de
    composición que draw_center/draw_smoke). No recibe `mask`: nunca se marca
    como clase propia, queda como fondo salvo que humo/trayectoria/derrumbe lo
    cubran después (ver docstring del módulo)."""
    if rng.random() >= prob:
        return

    h, w = tensor.shape
    brightness = _terrain_blotch_brightness(h, w, rng)
    tensor[:] = np.maximum(tensor, brightness.astype(np.uint8))
