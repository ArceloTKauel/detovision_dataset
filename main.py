"""
main.py - Punto de entrada del generador de dataset sintético de explosiones.

Orquesta el pipeline completo de generación: crea el lienzo, genera la zona de
impacto (cuadrilátero), distribuye centros de metralla, dibuja el humo con
textura Perlin, traza trayectorias rectas y parabólicas, y exporta el resultado
como imagen PNG en escala de grises binarizada (blanco/negro).

Funciones:
    - generate_explosion(height, width, rng): Genera un tensor 2D con una
      explosión sintética completa. Retorna un np.ndarray binarizado (0 o 255).
    - main(): Genera 4 imágenes de ejemplo y las guarda como explosion_N.png.
"""

import numpy as np

from canvas import (
    create_canvas,
    generate_quadrilateral,
    centroid_of_polygon,
    draw_center,
    distribute_centers_in_quadrilateral,
)
from smoke import draw_smoke
from trajectories import draw_trajectories
from export import tensor_to_image, mask_to_rgb

HEIGHT = 720
WIDTH = 1280


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (tensor B/W, mask de segmentación) generados en un solo pase."""
    if rng is None:
        rng = np.random.default_rng()

    # Lienzo vacío en escala de grises + máscara de segmentación (0=fondo, 1=humo, 2=trayectoria)
    tensor = create_canvas(height, width)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Zona de impacto: cuadrilátero aleatorio con su centroide como origen
    quad = generate_quadrilateral(height, width, rng)
    origin = centroid_of_polygon(quad)

    # Centro principal de la explosión
    center_size = rng.integers(1, 3)
    draw_center(tensor, origin, center_size, mask)

    # Centros secundarios distribuidos dentro del cuadrilátero (simulan fragmentos)
    num_centers = rng.integers(40, 80)
    centers = distribute_centers_in_quadrilateral(quad, num_centers, rng)

    for center in centers:
        draw_center(tensor, center, rng.integers(0, 2), mask)

    # Humo: radio proporcional al tamaño del lienzo
    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    smoke_radius = base_length * rng.uniform(0.15, 0.3)
    draw_smoke(tensor, centers, smoke_radius, rng, mask)

    # Ángulo del dron: apunta hacia el centro del lienzo con perturbación aleatoria.
    # Influye en la curvatura de las trayectorias parabólicas.
    drone_angle = np.arctan2(height / 2 - origin[0], width / 2 - origin[1])
    drone_angle += rng.uniform(-np.pi / 6, np.pi / 6)

    # Trayectorias de metralla (rectas + parabólicas)
    num_straight = rng.integers(1, 16)
    num_parabolic = rng.integers(1, 16)
    draw_trajectories(tensor, centers, origin, num_straight, num_parabolic, drone_angle, rng, mask)

    # Binarización: valores >= 128 pasan a blanco (255), el resto a negro (0)
    tensor = np.where(tensor >= 128, 255, 0).astype(np.uint8)

    return tensor, mask


def main():
    for i in range(1, 5):
        tensor, mask = generate_explosion(HEIGHT, WIDTH)
        tensor_to_image(tensor, f"explosion_{i}.png")
        mask_to_rgb(mask, f"explosion_{i}_mask.png")


if __name__ == "__main__":
    main()
