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

# Probabilidad de que una imagen del dataset lleve manchón de terreno. Antes era
# 0.5, igual que FAKE_TERRAIN_PROB en detovision_segmentation/utils/dataset.py,
# pero acá la decisión queda fija para siempre en esa imagen (no se resortea por
# época).
#
# Pasó a 1.0 con el dataset temporal (ver sequence.py). Con 0.5, una explosión
# sin terreno producía frames pre-ignición sin absolutamente nada: medido sobre
# un lote de 50 secuencias, el 23% de los frames salía completamente negro y el
# 56% de las secuencias tenía al menos uno. En una imagen suelta eso pasaba
# desapercibido porque la explosión estaba igual; en una secuencia son una docena
# de entradas vacías idénticas.
#
# Y no existen en la realidad: en las siete referencias el terreno llena el
# cuadro desde el primer frame, con p50 entre 7 y 10. Un frame vacío no es un
# caso difícil, es una entrada sin información que el modelo nunca va a ver en
# producción.
TERRAIN_PROB = 1.0

# ── Campo de "elevación" (dirección/curvatura del parallax) ────────────────
TERRAIN_FIELD_OCTAVES    = (1, 2)
TERRAIN_FIELD_CELL_RANGE = (250.0, 500.0)
TERRAIN_PERTURB_STRENGTH = (0.01, 0.05)
TERRAIN_VANISHING_MARGIN = 1.5

# ── Manchón: el campo se usa directo como brillo, modulado en bandas suaves ─
TERRAIN_BAND_PERIOD   = (0.04, 0.12)
TERRAIN_BLOTCH_MAX_VAL = (15, 55)
TERRAIN_BLUR_RADIUS   = (0.3, 0.8)

# Piso de la banda (fracción del período que queda "apagada"): en vez de un
# escalar único por imagen, se modula con un campo de baja frecuencia para
# que convivan zonas de banda ancha (floor bajo) y zonas de banda fina (floor
# alto) dentro del mismo cuadro, sin cambiar la forma/espaciado de las ondas.
#
# El rango está calibrado contra las máscaras reales: medido como p95 de 2*EDT
# sobre los píxeles encendidos y a 768x512 (el tamaño que ve el modelo), las
# referencias dan ~19 px de ancho de línea. Con (0.35, 0.90) salían 27.5 px,
# ~1.4x más anchas; este rango las deja en 19.5 px. Es el lever correcto para
# afinar porque no toca el período, o sea que la forma y el espaciado de las
# ondas quedan igual.
TERRAIN_BLOTCH_FLOOR_RANGE     = (0.65, 0.96)
TERRAIN_FLOOR_FIELD_OCTAVES    = (2, 3)
TERRAIN_FLOOR_FIELD_CELL_RANGE = (150.0, 320.0)
# Veces que se aplica smoothstep al campo: empuja los valores hacia los
# extremos (zona fina / zona ancha bien diferenciadas) manteniendo la
# transición suave entre ambas, en vez de quedarse en grises intermedios.
TERRAIN_FLOOR_FIELD_CONTRAST_PASSES = 2

# Grano fibroso de alta frecuencia, multiplicado sobre la banda.
TERRAIN_GRAIN_OCTAVES    = (3, 5)
TERRAIN_GRAIN_CELL_RANGE = (4.0, 12.0)
TERRAIN_GRAIN_CONTRAST   = (0.5, 0.9)

# ── Cortes en las bandas finas ─────────────────────────────────────────────
# Las líneas finas de las referencias reales son discontinuas, no un trazo
# continuo. Un campo de ruido aparte borra parte de la banda; el umbral se
# escala por el mismo campo de floor, así que las zonas anchas (floor bajo)
# quedan intactas y solo se cortan las finas. El grano fibroso no sirve para
# esto: promedia varias octavas, se concentra cerca de 0.5 y nunca llega a
# valores lo bastante bajos como para apagar un píxel.
#
# Qué banda se puede cortar se decide por su ancho REAL en píxeles, no por el
# campo de floor: `floor_norm` se normaliza por imagen, así que siempre hay un
# tercio "más fino que el resto" aunque en píxeles siga siendo ancho — el ancho
# real depende también del período de banda. Cortar por floor deja agujeros
# negros en medio de bandas anchas, que no se leen como línea discontinua sino
# como defecto. La apertura morfológica sí mide ancho absoluto: una banda más
# gruesa que ~2*RADIUS+1 px sobrevive intacta (con sus bordes) y queda
# protegida, mientras que una fina desaparece en la erosión y es cortable.
TERRAIN_DASH_STRENGTH    = 0.65
TERRAIN_DASH_THIN_RADIUS = 3
TERRAIN_DASH_LIT_LEVEL   = 0.15
TERRAIN_DASH_OCTAVES     = (2, 3)
TERRAIN_DASH_CELL_RANGE  = (5.0, 16.0)

# Intensidad global por muestra.
TERRAIN_INTENSITY_RANGE = (0.05, 1.0)


def _multi_octave_field(h, w, octaves_range, cell_range, rng, resample=Image.BILINEAR):
    """Campo [h, w] de ruido fractal multi-octava — misma técnica que
    _multi_octave_field en detovision_segmentation/utils/dataset.py, portada a
    `rng.random` en vez de `np.random.rand` (estado global) para que la
    generación quede determinada por el seed por índice de generate_dataset.py.

    `resample` permite pedir BICUBIC cuando la grilla base es muy chica (pocas
    celdas): con BILINEAR el estirado deja costuras rectas horizontales y
    verticales en los bordes de celda, visibles como quiebres antinaturales."""
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
    """Máscara booleana de píxeles a apagar para cortar en segmentos las bandas
    finas, dejando intactas las anchas — ver TERRAIN_DASH_STRENGTH.

    La apertura (erosión seguida de dilatación) se hace repitiendo MinFilter/
    MaxFilter de 3x3: equivale a un kernel de 2*RADIUS+1 y es bastante más
    rápido que pedirle a PIL ese kernel de una."""
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
    """Campo [h, w] en [0, 1] que module el piso de banda espacialmente —
    valor alto en una zona da bandas finas ahí, valor bajo da bandas más
    anchas, para que ambos "tipos" convivan dentro de la misma imagen.

    Usa BICUBIC (la grilla base son pocas celdas, ver _multi_octave_field) y
    remata con smoothstep para que las dos zonas queden bien diferenciadas en
    vez de promediarse en un gris intermedio."""
    field = _multi_octave_field(h, w, TERRAIN_FLOOR_FIELD_OCTAVES, TERRAIN_FLOOR_FIELD_CELL_RANGE,
                                rng, resample=Image.BICUBIC)
    field = _stretch_to_unit_range(field)
    for _ in range(TERRAIN_FLOOR_FIELD_CONTRAST_PASSES):
        field = field * field * (3.0 - 2.0 * field)
    return field


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
