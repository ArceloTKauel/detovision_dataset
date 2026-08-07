"""
export.py - Exportación de tensores a imágenes PNG.

Convierte los arrays numpy a imágenes PNG usando Pillow.

Funciones:
    - tensor_to_image(tensor, path): Guarda un tensor 2D en escala de grises.
    - contact_sheet(paths, path, brightness): Junta varias imágenes numeradas en
      una grilla, para revisar una secuencia temporal completa de un vistazo.
    - mask_to_rgb(mask, heatmap, path): Guarda la máscara de segmentación como
      PNG RGB. Fondo/humo/derrumbe se pintan con color plano (rojo/verde/
      amarillo); la clase trayectoria usa el canal azul con la intensidad del
      heatmap (gradiente: intenso al centro, tenue hacia el borde). No usa
      modo paleta ("P") porque una paleta de 256 entradas no puede representar
      un gradiente continuo.
"""

import numpy as np
from PIL import Image, ImageDraw

CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_TILE = (384, 256)
_LABEL_HEIGHT = 18


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)


def contact_sheet(paths: list[str], path: str, brightness: float = 1.0) -> None:
    """Grilla con las imágenes de `paths` en orden, cada una rotulada con su
    índice. `brightness` multiplica el resultado: las imágenes de este dominio
    son muy oscuras (las referencias reales tienen p50 entre 7 y 10 sobre 255) y
    sin realzar no se distingue nada a tamaño de miniatura.
    """
    tiles = [Image.open(p).convert("RGB") for p in paths]
    tw, th = CONTACT_SHEET_TILE
    cols = CONTACT_SHEET_COLUMNS
    rows = (len(tiles) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * tw, rows * (th + _LABEL_HEIGHT)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        y = row * (th + _LABEL_HEIGHT)
        sheet.paste(tile.resize((tw, th), Image.BOX), (col * tw, y))
        draw.text((col * tw + 6, y + th + 3), str(i), fill=(255, 255, 255))

    if brightness != 1.0:
        boosted = np.clip(np.asarray(sheet).astype(np.float32) * brightness, 0, 255)
        sheet = Image.fromarray(boosted.astype(np.uint8))
    sheet.save(path)


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
