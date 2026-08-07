#
# Generación masiva del dataset temporal.
#
# A diferencia de generate_dataset.py, que produce una imagen final por
# explosión, acá cada explosión produce una SECUENCIA completa: del terreno
# vacío pre-explosión hasta la vista final. Ver sequence.py para el porqué y
# para la semántica exacta (acumulación monótona, nada se apaga nunca).
#
# Estructura de salida, una carpeta por secuencia dentro de inputs/ y targets/:
#
#     dataset_sequences/
#         inputs/00000/000.png, 001.png, ...   → escala de grises
#         targets/00000/000.png, 001.png, ...  → máscaras RGB
#
# Se mantiene inputs/ y targets/ arriba, como en generate_dataset.py, para que
# el repo hermano no tenga que cambiar de convención: lo único nuevo es el nivel
# de carpeta por secuencia.
#
# El último frame de cada secuencia es exactamente la imagen que produciría
# generate_dataset.py con la misma semilla, así que un dataset de secuencias
# contiene al de imagen única como caso particular.

import os
from multiprocessing import Pool
from tqdm import tqdm

from main import HEIGHT, WIDTH
from sequence import generate_explosion_sequence
from export import tensor_to_image, mask_to_rgb

# Secuencias, NO imágenes: cada una produce ~29 frames, así que son ~87.000 pares
# imagen/máscara (~5.7 GB). El dataset de imagen única tenía 10.000 explosiones en
# 10.000 imágenes; acá lo que se mantiene alto a propósito es la variedad de
# EXPLOSIONES, porque los frames de una misma secuencia están muy correlacionados
# entre sí y aportan poco de a uno. Bajar esto a 346 para conservar "10.000
# imágenes" dejaría al modelo viendo el mismo terreno 29 veces.
#
# 3.000 y no 10.000 porque una época escala con el total de frames: con 10.000
# secuencias cada época tarda 29x más que en v20, y no vale comprometerse a eso
# antes de saber si el enfoque temporal sirve. Escalar después es regenerar.
TOTAL_SEQUENCES = 3_000
DATASET_DIR = "dataset_sequences"

# Desfase entre la semilla de la explosión y la de su estructura temporal. Son
# dos streams separados (ver generate_explosion_sequence): con el mismo índice
# en ambos, dos secuencias distintas compartirían el sorteo temporal.
_TIME_SEED_OFFSET = 10_000_000


def generate_single(index: int) -> int:
    """Worker: genera una secuencia completa y la guarda a disco."""
    import numpy as np

    frames = generate_explosion_sequence(
        HEIGHT, WIDTH,
        np.random.default_rng(index),
        np.random.default_rng(_TIME_SEED_OFFSET + index),
    )

    name = f"{index:05d}"
    input_dir = os.path.join(DATASET_DIR, "inputs", name)
    target_dir = os.path.join(DATASET_DIR, "targets", name)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    for i, (tensor, mask, heatmap) in enumerate(frames):
        filename = f"{i:03d}.png"
        tensor_to_image(tensor, os.path.join(input_dir, filename))
        mask_to_rgb(mask, heatmap, os.path.join(target_dir, filename))

    return len(frames)


def main():
    os.makedirs(os.path.join(DATASET_DIR, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "targets"), exist_ok=True)

    workers = max(1, os.cpu_count() - 2)
    print(f"Generando {TOTAL_SEQUENCES} secuencias con {workers} workers...")

    total_frames = 0
    with Pool(processes=workers) as pool:
        for num_frames in tqdm(pool.imap_unordered(generate_single, range(TOTAL_SEQUENCES)),
                               total=TOTAL_SEQUENCES, desc="Generando", unit="seq"):
            total_frames += num_frames

    print(f"Dataset generado: {TOTAL_SEQUENCES} secuencias, {total_frames} frames "
          f"({total_frames / TOTAL_SEQUENCES:.1f} por secuencia).")


if __name__ == "__main__":
    main()
