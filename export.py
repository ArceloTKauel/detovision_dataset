"""
export.py - Exportación de tensores a imágenes PNG.

Convierte los arrays numpy (escala de grises, uint8) a imágenes PNG
usando Pillow.

Funciones:
    - tensor_to_image(tensor, path): Guarda el tensor como imagen en modo "L"
      (escala de grises) en la ruta especificada.
"""

import numpy as np
from PIL import Image


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)
    print(f"Imagen guardada en: {path}")
