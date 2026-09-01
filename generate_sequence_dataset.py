"""
generate_sequence_dataset.py - Generación masiva del dataset temporal.

A diferencia de generate_dataset.py, cada explosión produce una SECUENCIA por
ventanas: cada frame muestra solo lo que nace en su tramo, no el acumulado (ver
sequence.py).

La unidad de entrenamiento es un BLOQUE de BLOCK_SIZE frames consecutivos, que el
modelo recibe apilados como canales — (batch, 9, alto, ancho) — y devuelve UNA
máscara POR FRAME: (batch, 9, 3, alto, ancho).

Estructura de salida, misma ruta relativa en los dos lados:
    dataset_sequences/inputs/00000_00/000.png ... 008.png   9 frames, escala de grises
    dataset_sequences/targets/00000_00/000.png ... 008.png  9 máscaras RGB

La correspondencia es 1 a 1: `targets/X/00k.png` es la máscara de `inputs/X/00k.png`.

Antes el target era UNO por bloque, la unión de los 9 frames (`_block_union`, ganaba
el evento más tardío). El bloque sigue entrando junto —los 9 frames son el contexto
temporal que necesita para distinguir un evento del terreno— pero ahora se le pide
segmentar cada uno, no resumirlos: la unión perdía CUÁNDO pasó cada cosa, que es
justo la información que el formato temporal vino a agregar.
"""

import os
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np

from main import HEIGHT, WIDTH
from sequence import generate_explosion_sequence
from export import tensor_to_image, mask_to_rgb

# Secuencias, NO imágenes: cada una da 90 frames y de ahí salen 10 bloques. Lo que
# se mantiene alto a propósito es la variedad de EXPLOSIONES, porque los frames de
# una misma secuencia están muy correlacionados.
#
# A 44.3 KB por par, 3.000 secuencias son 270.000 pares y ~12 GB. Conviene medir el
# tiempo por época antes de comprometerse: 500 secuencias son 2 GB y alcanzan para
# saber si el formato sirve.
TOTAL_SEQUENCES = 3_000                      # explosiones; cada una da varios bloques
DATASET_DIR = "dataset_sequences"            # carpeta de salida, con inputs/ y targets/

# NUM_FRAMES_RANGE en sequence.py es múltiplo de esto a propósito, así ninguna
# secuencia deja frames colgando sin bloque.
BLOCK_SIZE = 9                               # frames apilados como canales

# La explosión y su estructura temporal usan dos streams separados; con el mismo
# índice en ambos, dos secuencias distintas compartirían el sorteo temporal.
_TIME_SEED_OFFSET = 10_000_000               # separa el stream temporal del de la explosión


def generate_single(index: int) -> int:
    """Worker: genera una secuencia, la corta en bloques y los guarda a disco.

    Cada frame se escribe con su propia máscara y el mismo nombre relativo, así que
    emparejar entrada y target del lado del modelo es cambiar `inputs` por
    `targets` en la ruta.
    """
    frames = generate_explosion_sequence(
        HEIGHT, WIDTH,
        np.random.default_rng(index),
        np.random.default_rng(_TIME_SEED_OFFSET + index),
    )

    blocks = len(frames) // BLOCK_SIZE
    for b in range(blocks):
        block = frames[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        name = f"{index:05d}_{b:02d}"

        input_dir = os.path.join(DATASET_DIR, "inputs", name)
        target_dir = os.path.join(DATASET_DIR, "targets", name)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        for i, (tensor, mask, heatmap) in enumerate(block):
            tensor_to_image(tensor, os.path.join(input_dir, f"{i:03d}.png"))
            mask_to_rgb(mask, heatmap, os.path.join(target_dir, f"{i:03d}.png"))

    return blocks


def main():
    """Genera TOTAL_SEQUENCES secuencias en paralelo y reporta el avance."""
    os.makedirs(os.path.join(DATASET_DIR, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "targets"), exist_ok=True)

    workers = max(1, os.cpu_count() - 2)
    print(f"Generando {TOTAL_SEQUENCES} secuencias de bloques de {BLOCK_SIZE} "
          f"con {workers} workers...")

    total_blocks = 0
    with Pool(processes=workers) as pool:
        for blocks in tqdm(pool.imap_unordered(generate_single, range(TOTAL_SEQUENCES)),
                           total=TOTAL_SEQUENCES, desc="Generando", unit="seq"):
            total_blocks += blocks

    print(f"Dataset generado: {TOTAL_SEQUENCES} secuencias, {total_blocks} bloques, "
          f"{total_blocks * BLOCK_SIZE} imágenes de entrada.")


if __name__ == "__main__":
    main()
