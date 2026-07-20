"""
pixel_histogram.py - Compara la distribución de intensidad de píxeles entre
distintas clases (ej. trayectoria vs fondo) tomando un kernel NxN alrededor
de coordenadas dadas y graficando un histograma superpuesto por etiqueta.

Uso:
    uv run pixel_histogram.py <imagen> --punto X Y trayectoria --punto X Y fondo [--punto X Y fondo ...]

Cada --punto agrega una muestra de kernel×kernel píxeles (por defecto 5x5)
centrada en (X, Y) a la etiqueta indicada. Puntos con la misma etiqueta
acumulan sus píxeles en una sola distribución antes de graficar.
"""

import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEFAULT_KERNEL = 5
DEFAULT_OUT = "pixel_histogram.png"


def extract_patch(img: np.ndarray, x: int, y: int, kernel: int) -> np.ndarray:
    h, w = img.shape
    half = kernel // 2
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    return img[y0:y1, x0:x1].flatten()


def main():
    parser = argparse.ArgumentParser(description="Histograma comparativo de píxeles por clase (kernel NxN alrededor de coordenadas).")
    parser.add_argument("image", help="Ruta de la imagen a analizar")
    parser.add_argument(
        "--punto", action="append", nargs=3, metavar=("X", "Y", "ETIQUETA"),
        help="Coordenada (X Y) + etiqueta de clase. Repetible.",
    )
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL, help="Tamaño del kernel NxN (por defecto: %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Archivo de salida del histograma (por defecto: %(default)s)")
    args = parser.parse_args()

    if not args.punto:
        parser.error("hay que pasar al menos un --punto X Y ETIQUETA")

    img = np.array(Image.open(args.image).convert("L"))

    samples: dict[str, list[np.ndarray]] = {}
    for x_str, y_str, label in args.punto:
        x, y = int(x_str), int(y_str)
        patch = extract_patch(img, x, y, args.kernel)
        samples.setdefault(label, []).append(patch)

    print(f"Imagen: {args.image}  (kernel {args.kernel}x{args.kernel})")
    plt.figure(figsize=(8, 5))
    for label, patches in samples.items():
        values = np.concatenate(patches)
        print(f"  {label}: n={values.size}  media={values.mean():.2f}  std={values.std():.2f}  min={values.min()}  max={values.max()}")
        plt.hist(values, bins=range(0, 257, 4), alpha=0.55, label=f"{label} (n={values.size})")

    plt.xlabel("Intensidad de píxel (0-255)")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de intensidad por clase — kernel {args.kernel}x{args.kernel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()
