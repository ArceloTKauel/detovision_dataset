"""
export.py - Exportación de tensores a imágenes PNG.

Convierte los arrays numpy a imágenes PNG usando Pillow.

Funciones:
    - tensor_to_image(tensor, path): Guarda un tensor 2D en escala de grises.
    - mask_to_rgb(mask, heatmap, path): Guarda la máscara de segmentación como
      PNG RGB. Fondo/humo/derrumbe se pintan con color plano (rojo/verde/
      amarillo); la clase trayectoria usa el canal azul con la intensidad del
      heatmap (gradiente: intenso al centro, tenue hacia el borde). No usa
      modo paleta ("P") porque una paleta de 256 entradas no puede representar
      un gradiente continuo.
"""

import numpy as np
from PIL import Image


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)


def mask_to_rgb(mask: np.ndarray, heatmap: np.ndarray, path: str) -> None:
    """Combina la máscara categórica (fondo/humo/derrumbe) con el heatmap de
    gradiente de trayectoria en un único PNG RGB, y lo guarda en path."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 0] = (255, 0, 0)    # fondo -> rojo
    rgb[mask == 1] = (0, 255, 0)    # humo -> verde
    rgb[mask == 3] = (255, 255, 0)  # derrumbe -> amarillo

    # Trayectoria: azul con intensidad = heatmap. El heatmap ya respeta la
    # prioridad de clases (nunca se pinta sobre humo/derrumbe), así que no
    # hace falta forzar el orden acá.
    traj = heatmap > 0
    rgb[traj, 0] = 0
    rgb[traj, 1] = 0
    rgb[traj, 2] = heatmap[traj]

    Image.fromarray(rgb, mode="RGB").save(path)
