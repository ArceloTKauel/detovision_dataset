"""
generate_dataset.py - Generación masiva del dataset sintético.

Genera N pares de imágenes (entrada B/W + máscara RGB) usando multiprocessing
para aprovechar todos los cores disponibles de la CPU.

Estructura de salida:
    dataset/
        inputs/     → imágenes B/W binarizadas (entrada del modelo)
        targets/    → máscaras RGB de segmentación (salida esperada)
"""

import os
from multiprocessing import Pool

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb

TOTAL_IMAGES = 10_000
DATASET_DIR = "dataset"


def generate_single(index: int) -> int:
    """Worker: genera un par entrada/salida y lo guarda a disco."""
    import numpy as np

    rng = np.random.default_rng(seed=index)

    tensor, mask = generate_explosion(HEIGHT, WIDTH, rng)

    filename = f"{index:05d}.png"
    tensor_to_image(tensor, os.path.join(DATASET_DIR, "inputs", filename))
    mask_to_rgb(mask, os.path.join(DATASET_DIR, "targets", filename))

    return index


def main():
    os.makedirs(os.path.join(DATASET_DIR, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "targets"), exist_ok=True)

    workers = max(1, os.cpu_count() - 2)
    print(f"Generando {TOTAL_IMAGES} pares de imágenes con {workers} workers...")

    completed = 0
    with Pool(processes=workers) as pool:
        for index in pool.imap_unordered(generate_single, range(TOTAL_IMAGES)):
            completed += 1
            if completed % 100 == 0 or completed == TOTAL_IMAGES:
                print(f"  Progreso: {completed}/{TOTAL_IMAGES} ({completed * 100 // TOTAL_IMAGES}%)")

    print("Dataset generado exitosamente.")


if __name__ == "__main__":
    main()
