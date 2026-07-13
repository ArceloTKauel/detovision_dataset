"""
main.py - Punto de entrada del generador de dataset sintético de explosiones.

Orquesta el pipeline completo de generación en un solo pase: crea el lienzo,
genera la zona de impacto (cuadrilátero), distribuye centros de metralla,
dibuja el humo con textura Perlin, traza franjas de derrumbe (opcional) y
trayectorias rectas y parabólicas, y produce simultáneamente dos salidas:
    - Imagen en escala de grises (entrada del dataset): el humo queda en
      gradiente continuo (más intenso cerca del centro de la explosión, más
      tenue hacia el borde). Las trayectorias por ahora siguen dibujándose
      a blanco pleno (255); su propio gradiente es un cambio pendiente.
    - Máscara en PNG RGB (salida del dataset): fondo/humo/derrumbe con color
      plano (rojo/verde/amarillo); la trayectoria con gradiente real de azul
      (intenso al centro, tenue hacia el borde), vía el canal heatmap.

Funciones:
    - generate_explosion(height, width, rng): Genera la explosión completa.
      Retorna una tupla (tensor B/W, mask categórica, heatmap de trayectoria).
    - main(): Genera 4 pares de imágenes (explosion_N.png + explosion_N_mask.png).
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
from landslide import draw_landslides
from trajectories import draw_trajectories
from export import tensor_to_image, mask_to_rgb

HEIGHT = 512
WIDTH = 768
DRAW_LANDSLIDES = False


def generate_explosion(height: int, width: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (tensor B/W, mask categórica, heatmap de trayectoria), generados en un solo pase."""
    if rng is None:
        rng = np.random.default_rng()

    # Lienzo vacío en escala de grises + máscara de segmentación (0=fondo, 1=humo, 2=trayectoria)
    # + heatmap de gradiente para la clase trayectoria (0 = sin trayectoria, 255 = núcleo)
    tensor = create_canvas(height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    heatmap = np.zeros((height, width), dtype=np.uint8)

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

    # Trayectorias de metralla (rectas + parabólicas de ida y vuelta)
    num_straight = rng.integers(15, 30)
    num_parabolic = rng.integers(15, 30)
    draw_trajectories(tensor, centers, origin, num_straight, num_parabolic, rng, mask, heatmap)

    # Derrumbe: franjas de desprendimiento independientes de la explosión,
    # aproximadamente paralelas entre sí, con puntos de inicio propios
    # distribuidos en cualquier parte del lienzo. Ocurre en ~40% de las imágenes.
    # Se dibuja al final para que respete la prioridad humo > trayectoria >
    # derrumbe > fondo, y nunca pasa por encima de la zona de la explosión
    # (exclusion_radius cubre el centro + los fragmentos + el humo).
    # Desactivado temporalmente: DRAW_LANDSLIDES = False deshabilita la clase 3.
    if DRAW_LANDSLIDES and rng.random() < 0.4:
        centers_arr = np.array(centers, dtype=np.float64)
        max_center_dist = np.sqrt(((centers_arr - origin) ** 2).sum(axis=1)).max()
        exclusion_radius = max_center_dist + smoke_radius * 1.3 + 10
        num_stripes = int(np.clip(round(rng.normal(4.5, 1.5)), 1, 8))
        draw_landslides(tensor, num_stripes, np.radians(15), rng, mask, origin, exclusion_radius)

    # Sincronizar mask: píxeles de humo que quedaron en negro puro (p. ej.
    # manchas sustractivas que no dibujaron nada) vuelven a fondo. Las
    # trayectorias (clase 2) no se tocan.
    mask[(tensor == 0) & (mask == 1)] = 0

    return tensor, mask, heatmap


def main():
    for i in range(1, 5):
        tensor, mask, heatmap = generate_explosion(HEIGHT, WIDTH)
        tensor_to_image(tensor, f"explosion_{i}.png")
        mask_to_rgb(mask, heatmap, f"explosion_{i}_mask.png")


if __name__ == "__main__":
    main()
