"""
export.py - Exportación de tensores a imágenes PNG.

Convierte los arrays numpy a imágenes PNG usando Pillow.

Funciones:
    - tensor_to_image(tensor, path): Guarda un tensor 2D en escala de grises.
    - mask_to_rgb(mask, path): Guarda una máscara de segmentación (0=fondo,
      1=humo, 2=trayectoria, 3=derrumbe) como PNG modo paleta ("P"): cada
      píxel es el índice de clase directo, y la paleta es solo metadata de
      visualización embebida en el propio archivo (no parte del dato).
      Paleta: Rojo=fondo, Verde=humo, Azul=trayectoria, Amarillo=derrumbe.
"""

import numpy as np
from PIL import Image

_PALETTE = [
    255, 0, 0,     # 0: fondo -> rojo
    0, 255, 0,     # 1: humo -> verde
    0, 0, 255,     # 2: trayectoria -> azul
    255, 255, 0,   # 3: derrumbe -> amarillo
]
_PALETTE = _PALETTE + [0, 0, 0] * (256 - len(_PALETTE) // 3)


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)


def mask_to_rgb(mask: np.ndarray, path: str) -> None:
    """Guarda mask (índices de clase 0-3) como PNG modo paleta indexada."""
    image = Image.fromarray(mask, mode="P")
    image.putpalette(_PALETTE)
    image.save(path)
