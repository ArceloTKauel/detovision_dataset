"""
export.py - Exportación de tensores a PNG.

La máscara sale en modo RGB y no en paleta ("P") porque la clase trayectoria lleva
un gradiente continuo en el canal azul, y una paleta de 256 entradas no puede
representarlo.
"""

import numpy as np
from PIL import Image, ImageDraw

CONTACT_SHEET_COLUMNS = 4                    # columnas de la grilla
CONTACT_SHEET_TILE = (384, 256)              # tamaño de cada miniatura, en px
_LABEL_HEIGHT = 18                           # alto de la banda con el número, en px


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    """Guarda un tensor 2D uint8 como PNG en escala de grises."""
    image = Image.fromarray(tensor, mode="L")
    image.save(path)


def contact_sheet(paths: list[str], path: str, brightness: float = 1.0) -> None:
    """Grilla con las imágenes de `paths` en orden, rotuladas con su índice, para
    revisar una secuencia completa de un vistazo. `brightness` multiplica el
    resultado: el dominio es muy oscuro (p50 entre 7 y 10) y sin realzar no se
    distingue nada a tamaño de miniatura."""
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
    """Combina la máscara categórica con el heatmap de trayectoria en un PNG RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 0] = (255, 0, 0)    # fondo -> rojo
    rgb[mask == 1] = (0, 255, 0)    # humo -> verde
    rgb[mask == 3] = (255, 255, 0)  # derrumbe -> amarillo

    # Trayectoria: azul con intensidad = heatmap. El heatmap ya respeta la
    # prioridad de clases, así que no hace falta forzar el orden acá.
    traj = heatmap > 0
    rgb[traj, 0] = 0
    rgb[traj, 1] = 0
    rgb[traj, 2] = heatmap[traj]

    Image.fromarray(rgb, mode="RGB").save(path)
