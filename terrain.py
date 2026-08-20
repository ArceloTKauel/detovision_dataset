"""
terrain.py - Manchón de parallax de terreno (ego-motion de cámara).

La textura de fondo que aparece en video real al diferenciar frames consecutivos,
por el movimiento de la cámara sobre el relieve. El modelo la confundía con la
clase trayectoria (ver mascara_cambios_final_sinbin_*.png en el repo hermano). Es
un gradiente CONTINUO de todo el cuadro, no una forma discreta como el humo.

Se dibuja primero y no toca `mask`: queda como fondo salvo donde otro elemento lo
cubra, que es lo correcto — es un artefacto de cámara, no un objeto a segmentar.

Misma técnica que la familia FAKE_TERRAIN_* de detovision_segmentation, que allá
sigue activa como augmentation por época; acá está horneada en la generación y
fija por imagen. Las dos capas conviven.
"""

import numpy as np
from PIL import Image, ImageFilter

# Pasó de 0.5 a 1.0 con el dataset temporal: sin terreno, el 23% de los frames
# pre-ignición salía negro y el 56% de las secuencias tenía alguno. En las siete
# referencias el terreno llena el cuadro desde el primer frame (p50 entre 7 y 10),
# así que un frame vacío no es un caso difícil sino una entrada sin información.
TERRAIN_PROB = 1.0                           # probabilidad de que la imagen lleve terreno

# ── Campo de "elevación" (dirección/curvatura del parallax) ────────────────
TERRAIN_FIELD_OCTAVES    = (1, 2)            # campo de elevación: octavas del ruido
TERRAIN_FIELD_CELL_RANGE = (250.0, 500.0)    # tamaño de celda base, en px
TERRAIN_PERTURB_STRENGTH = (0.01, 0.05)      # cuánto rompe la simetría del campo radial
TERRAIN_VANISHING_MARGIN = 1.5               # el punto de fuga puede caer fuera del cuadro

# ── Manchón: el campo se usa directo como brillo, modulado en bandas suaves ─
TERRAIN_BAND_PERIOD   = (0.04, 0.12)         # período de las bandas, en fracción del campo
TERRAIN_BLOTCH_MAX_VAL = (15, 55)            # brillo máximo del manchón, antes de intensity
TERRAIN_BLUR_RADIUS   = (0.3, 0.8)           # suavizado final, en px

# Fracción del período que queda apagada. No es un escalar por imagen sino un
# campo de baja frecuencia, para que convivan zonas de banda ancha (piso bajo) y
# fina (piso alto) en el mismo cuadro sin tocar la forma ni el espaciado.
#
# Calibrado contra las máscaras reales: medido como p95 de 2*EDT sobre los píxeles
# encendidos a 768x512, las referencias dan ~19 px de ancho de línea. Con
# (0.35, 0.90) salían 27.5 px; este rango las deja en 19.5. Es el lever correcto
# para afinar el ancho porque no toca el período.
TERRAIN_BLOTCH_FLOOR_RANGE     = (0.65, 0.96)  # piso de banda: alto = línea fina
TERRAIN_FLOOR_FIELD_OCTAVES    = (2, 3)      # campo que modula ese piso
TERRAIN_FLOOR_FIELD_CELL_RANGE = (150.0, 320.0)  # su tamaño de celda, en px
# El smoothstep empuja los valores a los extremos, así las dos zonas quedan bien
# diferenciadas en vez de promediarse en un gris intermedio.
TERRAIN_FLOOR_FIELD_CONTRAST_PASSES = 2      # pasadas de smoothstep: separa fina de ancha

# Grano fibroso de alta frecuencia, multiplicado sobre la banda.
TERRAIN_GRAIN_OCTAVES    = (3, 5)            # grano fibroso: octavas
TERRAIN_GRAIN_CELL_RANGE = (4.0, 12.0)       # su tamaño de celda, en px
TERRAIN_GRAIN_CONTRAST   = (0.5, 0.9)        # cuánto modula la banda

# ── Cortes en las bandas finas ─────────────────────────────────────────────
# Las líneas finas de las referencias son discontinuas. Un campo de ruido aparte
# borra parte de la banda; el grano fibroso no sirve para esto, porque promedia
# varias octavas y nunca baja lo suficiente como para apagar un píxel.
#
# Qué banda se puede cortar se decide por su ancho REAL en píxeles, con una
# apertura morfológica, y no por el campo de floor: floor se normaliza por imagen,
# así que siempre hay un tercio "más fino que el resto" aunque en píxeles siga
# siendo ancho, y cortar ahí deja agujeros negros en medio de bandas anchas que se
# leen como defecto. Con la apertura, una banda más gruesa que ~2*RADIUS+1 px
# sobrevive intacta y queda protegida; una fina desaparece en la erosión.
TERRAIN_DASH_STRENGTH    = 0.65              # umbral de corte: más alto, más discontinua
TERRAIN_DASH_THIN_RADIUS = 3                 # apertura morfológica: qué banda es "fina"
TERRAIN_DASH_LIT_LEVEL   = 0.15              # desde qué nivel la banda cuenta como encendida
TERRAIN_DASH_OCTAVES     = (2, 3)            # campo que decide dónde cortar
TERRAIN_DASH_CELL_RANGE  = (5.0, 16.0)       # su tamaño de celda, en px

# Intensidad global por muestra.
TERRAIN_INTENSITY_RANGE = (0.05, 1.0)        # brillo global: varía 20x entre imágenes


def _multi_octave_field(h, w, octaves_range, cell_range, rng, resample=Image.BILINEAR):
    """Campo [h, w] de ruido fractal multi-octava. Igual que en el repo hermano,
    pero sobre `rng.random` y no `np.random.rand` (estado global), para que la
    generación quede determinada por la semilla de generate_dataset.py.

    `resample` permite pedir BICUBIC cuando la grilla base tiene pocas celdas: con
    BILINEAR el estirado deja costuras rectas en los bordes de celda."""
    octaves = rng.integers(*octaves_range, endpoint=True)
    base_cell = rng.uniform(*cell_range)

    field = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amplitude = 1.0, 0.0
    for octave in range(octaves):
        cell = max(2.0, base_cell / (2 ** octave))
        grid_h = max(2, round(h / cell))
        grid_w = max(2, round(w / cell))
        small = rng.random((grid_h, grid_w)).astype(np.float32)
        layer = np.array(Image.fromarray(small).resize((w, h), resample))
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


def _terrain_dash_cut(h, w, rng, banding):
    """Máscara de píxeles a apagar para cortar en segmentos las bandas finas,
    dejando intactas las anchas (ver TERRAIN_DASH_STRENGTH). La apertura se hace
    repitiendo MinFilter/MaxFilter de 3x3: equivale a un kernel de 2*RADIUS+1 y es
    bastante más rápido que pedirle a PIL ese kernel de una."""
    dash = _stretch_to_unit_range(
        _multi_octave_field(h, w, TERRAIN_DASH_OCTAVES, TERRAIN_DASH_CELL_RANGE, rng))

    lit = banding > TERRAIN_DASH_LIT_LEVEL
    opened = Image.fromarray((lit * 255).astype(np.uint8))
    for _ in range(TERRAIN_DASH_THIN_RADIUS):
        opened = opened.filter(ImageFilter.MinFilter(3))
    for _ in range(TERRAIN_DASH_THIN_RADIUS):
        opened = opened.filter(ImageFilter.MaxFilter(3))

    thin = lit & ~(np.array(opened) > 127)
    return (dash < TERRAIN_DASH_STRENGTH) & thin


def _terrain_floor_field(h, w, rng):
    """Campo [0, 1] que modula el piso de banda espacialmente: alto da bandas finas
    ahí, bajo las da anchas, para que los dos tipos convivan en la misma imagen.
    Ver TERRAIN_BLOTCH_FLOOR_RANGE."""
    field = _multi_octave_field(h, w, TERRAIN_FLOOR_FIELD_OCTAVES, TERRAIN_FLOOR_FIELD_CELL_RANGE,
                                rng, resample=Image.BICUBIC)
    field = _stretch_to_unit_range(field)
    for _ in range(TERRAIN_FLOOR_FIELD_CONTRAST_PASSES):
        field = field * field * (3.0 - 2.0 * field)
    return field


def _terrain_blotch_brightness(h, w, rng):
    """Mapa de brillo [0, 255]: bandas por modulación seno del campo de elevación,
    recortadas por el piso, moduladas por el grano y escaladas por la intensidad
    global. Ver _fake_terrain_blotch_brightness en el repo hermano."""
    field = _terrain_elevation_field(h, w, rng)
    lo, hi = field.min(), field.max()
    normalized = (field - lo) / (hi - lo + 1e-8)

    period = rng.uniform(*TERRAIN_BAND_PERIOD)
    phase = rng.uniform(0, 2 * np.pi)
    banding = 0.5 + 0.5 * np.sin(2 * np.pi * normalized / period + phase)

    floor_lo, floor_hi = TERRAIN_BLOTCH_FLOOR_RANGE
    floor_field = floor_lo + (floor_hi - floor_lo) * _terrain_floor_field(h, w, rng)
    banding = np.clip((banding - floor_field) / (1.0 - floor_field), 0.0, 1.0)

    banding = np.clip(banding * _terrain_grain_multiplier(h, w, rng), 0.0, 1.0)

    # Las bandas que quedaron finas se cortan en segmentos; las anchas no.
    banding[_terrain_dash_cut(h, w, rng, banding)] = 0.0

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
    """Sortea si esta imagen lleva manchón de terreno y lo dibuja con np.maximum,
    el mismo criterio de composición que el resto del pipeline."""
    if rng.random() >= prob:
        return

    h, w = tensor.shape
    brightness = _terrain_blotch_brightness(h, w, rng)
    tensor[:] = np.maximum(tensor, brightness.astype(np.uint8))
