"""
canvas.py - Lienzo y geometría base de la explosión.

La zona de impacto es una FILA de pozos —la línea de tiro—, no un punto ni una
región: la carga real de una voladura es lineal y por eso la pluma sale alargada
sobre ella.

Funciones:
    - create_canvas(height, width): tensor 2D de ceros.
    - generate_blast_line(height, width, rng, margin): polilínea con la fila de
      pozos cargados.
    - sample_on_line(line, count, rng): puntos sorteados sobre esa polilínea.
    - distribute_centers_along_line(line, num_points, rng): reparte los centros
      de fragmento a lo largo de la línea, con dispersión lateral.
    - draw_center(tensor, center, size, rng, mask): dibuja un pozo, con brillo
      por debajo del rango del núcleo de humo para que quede camuflado cuando
      draw_smoke se dibuje encima.
"""

import numpy as np


# ── Línea de tiro ──────────────────────────────────────────────────────────
# En una voladura real la carga es una FILA de pozos a lo largo de un banco, no
# un punto: la pluma sale alargada sobre esa línea y cada punto de ella emite su
# propio abanico. Se ve explícito en las referencias reales
# (detovision_segmentation/inference/inputs/mascara_cambios_final_sinbin_3.png,
# donde la línea de tiro aparece como un trazo brillante, y la _13, que es una
# hilera de bocanadas contiguas).
#
# Antes los centros se repartían dentro de un cuadrilátero, lo que daba un blob
# redondeado: elongación medida 1.99 contra 3.42 de las referencias.
BLAST_LINE_LENGTH_RATIO = (0.10, 0.28)   # fracción de min(alto, ancho)
BLAST_LINE_BOW_RATIO    = (-0.12, 0.12)  # curvatura: los bancos no son rectos
BLAST_LINE_JITTER_RATIO = 0.05           # dispersión lateral de los pozos
_BLAST_LINE_SAMPLES     = 48


def generate_blast_line(
    height: int, width: int, rng: np.random.Generator, margin: float = 0.15
) -> np.ndarray:
    """Polilínea [N, 2] en (y, x) con la fila de pozos cargados: un segmento con
    curvatura leve, modelado como Bézier cuadrática. El ángulo se sortea en
    [0, pi) y no en [0, 2pi) porque la línea no tiene sentido — recorrerla al
    revés da la misma línea."""
    cy = rng.uniform(height * margin, height * (1 - margin))
    cx = rng.uniform(width * margin, width * (1 - margin))
    length = min(height, width) * rng.uniform(*BLAST_LINE_LENGTH_RATIO)

    angle = rng.uniform(0, np.pi)
    dy, dx = np.sin(angle), np.cos(angle)
    p0 = np.array([cy - dy * length / 2, cx - dx * length / 2])
    p2 = np.array([cy + dy * length / 2, cx + dx * length / 2])

    # Punto medio desplazado perpendicularmente; el control de la Bézier se
    # despeja de B(0.5) = (p0 + 2*ctrl + p2) / 4 para que la curva pase por él.
    bow = length * rng.uniform(*BLAST_LINE_BOW_RATIO)
    mid = np.array([cy - dx * bow, cx + dy * bow])
    ctrl = 2.0 * mid - 0.5 * (p0 + p2)

    t = np.linspace(0.0, 1.0, _BLAST_LINE_SAMPLES)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2


def sample_on_line(line: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` puntos [count, 2] repartidos uniformemente a lo largo de la
    polilínea, interpolando dentro de cada tramo."""
    idx = rng.integers(0, len(line) - 1, size=count)
    frac = rng.random(count)[:, None]
    return line[idx] * (1 - frac) + line[idx + 1] * frac


def distribute_centers_along_line(
    line: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    jitter_ratio: float = BLAST_LINE_JITTER_RATIO,
) -> list[tuple[int, int]]:
    """Centros de fragmento repartidos sobre la línea de tiro, con dispersión
    lateral (perpendicular) mayor que la longitudinal — los pozos están
    alineados, lo que se dispersa es el material que arrojan."""
    pts = sample_on_line(line, num_points, rng)

    span_vec = line[-1] - line[0]
    span = float(np.linalg.norm(span_vec))
    direction = span_vec / (span + 1e-9)
    perp = np.array([-direction[1], direction[0]])

    pts = pts + rng.normal(0, span * jitter_ratio, size=num_points)[:, None] * perp
    pts = pts + rng.normal(0, span * jitter_ratio * 0.5, size=num_points)[:, None] * direction
    return [(int(round(p[0])), int(round(p[1]))) for p in pts]


def create_canvas(height: int, width: int) -> np.ndarray:
    """Lienzo vacío en escala de grises."""
    return np.zeros((height, width), dtype=np.uint8)


# Rango de brillo de los cuadrados de centro: deliberadamente por debajo del
# núcleo de humo (~217-255, ver smoke.py core_brightness) para que, una vez
# que draw_smoke se dibuja encima (después, con np.maximum), el cuadrado
# quede camuflado dentro de la textura del humo en vez de verse como un
# parche plano y perfectamente cuadrado.
_CENTER_BRIGHTNESS_RANGE = (120.0, 200.0)


def draw_center(
    tensor: np.ndarray,
    center: tuple[int, int],
    size: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
) -> None:
    """Dibuja un cuadrado de lado (2*size+1) píxeles, con brillo variable por píxel.

    brightness_scale: misma escala de brillo global sorteada para el humo
    (ver smoke.py::sample_brightness_scale), para que el rango de camuflaje
    siga quedando por debajo del núcleo de humo aunque este se oscurezca.
    """
    h, w = tensor.shape
    cy, cx = center
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                brightness = int(rng.uniform(*_CENTER_BRIGHTNESS_RANGE) * brightness_scale)
                tensor[ny, nx] = max(tensor[ny, nx], brightness)
                # Los centros son parte del humo en la máscara (clase 1)
                if mask is not None:
                    mask[ny, nx] = 1
