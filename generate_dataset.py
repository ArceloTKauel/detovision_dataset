"""
generate_dataset.py - Generación masiva del dataset de imagen única.

N pares (entrada en escala de grises + máscara RGB) con multiprocessing, uno por
semilla. Para el dataset temporal ver generate_sequence_dataset.py.

Estructura de salida:
    dataset/inputs/     imágenes en escala de grises
    dataset/targets/    máscaras RGB
"""

import os
from multiprocessing import Pool
from tqdm import tqdm

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb

TOTAL_IMAGES = 10_000                        # pares entrada/máscara a generar
DATASET_DIR = "dataset"                      # carpeta de salida, con inputs/ y targets/


def generate_single(index: int) -> int:
    """Worker: genera un par entrada/salida y lo guarda a disco."""
    import numpy as np

    rng = np.random.default_rng(seed=index)

    tensor, mask, heatmap = generate_explosion(HEIGHT, WIDTH, rng)

    filename = f"{index:05d}.png"
    tensor_to_image(tensor, os.path.join(DATASET_DIR, "inputs", filename))
    mask_to_rgb(mask, heatmap, os.path.join(DATASET_DIR, "targets", filename))

    return index


def main():
    """Genera TOTAL_IMAGES pares en paralelo y reporta el avance."""
    os.makedirs(os.path.join(DATASET_DIR, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "targets"), exist_ok=True)

    workers = max(1, os.cpu_count() - 2)
    print(f"Generando {TOTAL_IMAGES} pares de imágenes con {workers} workers...")

    with Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap_unordered(generate_single, range(TOTAL_IMAGES)), total=TOTAL_IMAGES, desc="Generando", unit="img"):
            pass

    print("Dataset generado exitosamente.")


if __name__ == "__main__":
    main()
