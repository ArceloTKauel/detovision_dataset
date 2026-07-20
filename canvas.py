"""
canvas.py - Creación del lienzo y geometría base de la explosión.

Genera el tensor vacío (escala de grises), crea un cuadrilátero aleatorio que
define la zona de impacto, calcula su centroide, y distribuye puntos dentro
del cuadrilátero que actúan como centros de fragmentos/metralla.

Funciones:
    - create_canvas(height, width): Crea un tensor 2D de ceros (fondo negro).
    - generate_quadrilateral(height, width, rng, margin): Genera 4 vértices
      aleatorios alrededor de un centro random, formando la zona de impacto.
    - centroid_of_polygon(vertices): Calcula el centroide promediando vértices.
    - draw_center(tensor, center, size, rng, mask): Dibuja un cuadrado en la
      posición dada, de lado (2*size+1), con brillo sorteado por debajo del
      rango del núcleo de humo (ver smoke.py) para que quede camuflado una
      vez que draw_smoke se dibuje encima. Si se pasa mask, marca los
      píxeles como humo (clase 1).
    - distribute_centers_in_quadrilateral(vertices, num_points, rng): Distribuye
      puntos aleatorios uniformemente dentro del cuadrilátero usando muestreo
      por triángulos.
"""

import numpy as np


def create_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def generate_quadrilateral(
    height: int, width: int, rng: np.random.Generator, margin: float = 0.15
) -> np.ndarray:
    # Centro del cuadrilátero: posición aleatoria respetando márgenes
    cx = rng.integers(int(width * margin), int(width * (1 - margin)))
    cy = rng.integers(int(height * margin), int(height * (1 - margin)))

    # 4 vértices a distancias y ángulos aleatorios desde el centro
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=4))
    radii = rng.uniform(20, 60, size=4)

    vertices = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        vertices[i, 0] = cy + radii[i] * np.sin(angles[i])  # coordenada Y
        vertices[i, 1] = cx + radii[i] * np.cos(angles[i])  # coordenada X

    return vertices


def centroid_of_polygon(vertices: np.ndarray) -> tuple[int, int]:
    cy = int(round(np.mean(vertices[:, 0])))
    cx = int(round(np.mean(vertices[:, 1])))
    return cy, cx


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


# Grano de fondo: calibrado contra los heatmaps de diferencia de video reales
# sin binarizar (detovision_segmentation/inference/inputs/mascara_cambios_final_sinbin_*.png),
# donde el fondo mide media ~6-15 y desvío ~1.5-5 por parche, sin apenas nada
# por encima de ~30-40, y 0.000% de los píxeles en negro puro (mínimo real
# observado: 1) en las 6 imágenes de referencia. Media y desvío se sortean
# una vez por imagen (no por píxel) para que la textura varíe de una
# explosión a otra, igual que brightness_scale en smoke.py.
_BACKGROUND_MEAN_RANGE = (15.0, 30.0)
_BACKGROUND_STD_RANGE = (5.0, 10.0)
_BACKGROUND_MAX = 40


def apply_background_noise(
    tensor: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Reemplaza cada píxel en negro puro (0) por ruido gaussiano (media/desvío
    sorteados una vez por imagen dentro de _BACKGROUND_MEAN_RANGE/_STD_RANGE,
    recortado a [1, _BACKGROUND_MAX] — el piso en 1, no 0, para que ningún
    píxel de fondo quede en negro puro, igual que en las imágenes reales de
    referencia), dándole al fondo grano de gris oscuro en vez de negro plano.
    Debe llamarse al final del pipeline, cuando ya no queda ninguna lógica
    que dependa de detectar 0 como "sin dibujar" (measure_smoke_width en
    smoke.py, sync de mask en main.py).
    """
    zero_mask = tensor == 0
    n = int(zero_mask.sum())
    if n:
        mean = rng.uniform(*_BACKGROUND_MEAN_RANGE)
        std = rng.uniform(*_BACKGROUND_STD_RANGE)
        noise = rng.normal(mean, std, size=n)
        tensor[zero_mask] = np.clip(noise, 1, _BACKGROUND_MAX).astype(np.uint8)


def distribute_centers_in_quadrilateral(
    vertices: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Muestreo uniforme dentro del cuadrilátero.
    Se divide en 2 triángulos (v0-v1-v2 y v0-v2-v3) y se elige uno al azar
    para cada punto. Dentro del triángulo se usa muestreo baricéntrico:
    si r1+r2 > 1 se reflejan para mantener uniformidad.
    """
    v0, v1, v2, v3 = vertices
    centers = []
    for _ in range(num_points):
        # Elegir triángulo al azar (50/50)
        if rng.random() < 0.5:
            tri = [v0, v1, v2]
        else:
            tri = [v0, v2, v3]

        # Muestreo baricéntrico uniforme
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        point = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
        centers.append((int(round(point[0])), int(round(point[1]))))
    return centers
