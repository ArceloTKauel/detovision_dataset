"""
zoom_preview.py - Genera un recorte ampliado de una imagen para revisar detalle
(gradientes, puntos, bordes) sin tener que abrir el archivo completo.

Uso:
    uv run zoom_preview.py <imagen> [--box x0 y0 x1 y1] [--zoom N] [--out archivo]

Por defecto recorta una zona de 150x100px cerca del centro típico de trayectorias
y la escala 6x con NEAREST (sin suavizado, para ver el píxel real).
"""

import argparse
import os

from PIL import Image

DEFAULT_BOX = (550, 50, 700, 150)
DEFAULT_ZOOM = 6


def make_zoom(image_path: str, box: tuple[int, int, int, int], zoom: int, out_path: str | None = None) -> str:
    """Recorta una ventana alrededor de (cy, cx) y la amplía por vecino más
    cercano, para inspeccionar píxeles sin que el remuestreo los mezcle."""
    img = Image.open(image_path)
    crop = img.crop(box)
    crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_zoom{ext}"

    crop.save(out_path)
    return out_path


def main():
    """Escribe los recortes ampliados de las imágenes indicadas por línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera un recorte ampliado de una imagen.")
    parser.add_argument("image", help="Ruta de la imagen a recortar")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                         default=DEFAULT_BOX, help="Región a recortar (por defecto: %(default)s)")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help="Factor de escala (por defecto: %(default)s)")
    parser.add_argument("--out", default=None, help="Ruta de salida (por defecto: <imagen>_zoom.png)")
    args = parser.parse_args()

    out_path = make_zoom(args.image, tuple(args.box), args.zoom, args.out)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
