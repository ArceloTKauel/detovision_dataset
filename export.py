"""
export.py - Exportación de tensores a imágenes PNG.

Convierte los arrays numpy a imágenes PNG usando Pillow.

Funciones:
    - tensor_to_image(tensor, path): Guarda un tensor 2D en escala de grises.
    - mask_to_rgb(mask, path): Convierte una máscara de segmentación (0=fondo,
      1=humo, 2=trayectoria) a imagen RGB y la guarda.
      Colores: Rojo=fondo, Verde=humo, Azul=trayectoria.
"""

import numpy as np
from PIL import Image


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)
    print(f"Imagen guardada en: {path}")


def mask_to_rgb(mask: np.ndarray, path: str) -> None:
    """Convierte mask de clases (0/1/2) a RGB: Rojo=fondo, Verde=humo, Azul=trayectoria."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[mask == 0] = [255, 0, 0]    # Fondo → Rojo
    rgb[mask == 1] = [0, 255, 0]    # Humo → Verde
    rgb[mask == 2] = [0, 0, 255]    # Trayectoria → Azul

    image = Image.fromarray(rgb, mode="RGB")
    image.save(path)
    print(f"Máscara guardada en: {path}")
