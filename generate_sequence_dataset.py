"""
generate_sequence_dataset.py - Generación masiva del dataset temporal.

A diferencia de generate_dataset.py, cada explosión produce una SECUENCIA por
ventanas: cada frame muestra solo lo que nace en su tramo, no el acumulado (ver
sequence.py).

La unidad de entrenamiento es un BLOQUE de BLOCK_SIZE frames consecutivos, que el
modelo recibe apilados como canales — (batch, 9, alto, ancho) — con UNA máscara
por bloque, la unión de lo que muestran sus frames.

Estructura de salida, mismo stem para que la correspondencia sea obvia:
    dataset_sequences/inputs/00000_00/000.png ... 008.png   9 canales
    dataset_sequences/targets/00000_00.png                  máscara RGB

Por qué el target es la unión del bloque y no el acumulado desde el principio:
cada bloque es una pasada independiente del modelo, que no vio los anteriores.
Pedirle el acumulado sería pedirle que recuerde algo que no está en su entrada;
si producción quiere un heatmap corrido, acumula las salidas del modelo.
"""

import os
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np

from main import HEIGHT, WIDTH
from sequence import generate_explosion_sequence
from export import tensor_to_image, mask_to_rgb

# Secuencias, NO imágenes: cada una produce NUM_FRAMES_RANGE frames (90, fijo) y
# de ahí salen 10 bloques. Lo que se mantiene alto a propósito es la variedad de
# EXPLOSIONES, porque los frames de una misma secuencia están muy correlacionados
# y aportan poco de a uno.
#
# A 44.3 KB por par: 3.000 secuencias son 270.000 pares y ~12 GB. Conviene medir
# el tiempo por época antes de comprometerse a ese tamaño — 500 secuencias son
# 45.000 pares y 2 GB, y alcanzan para saber si el formato sirve.
TOTAL_SEQUENCES = 3_000                      # explosiones; cada una da varios bloques
DATASET_DIR = "dataset_sequences"            # carpeta de salida, con inputs/ y targets/

# Frames que entran juntos como canales del tensor. NUM_FRAMES_RANGE en
# sequence.py es múltiplo de esto a propósito, así que ninguna secuencia deja
# frames colgando sin bloque.
BLOCK_SIZE = 9                               # frames apilados como canales

# Desfase entre la semilla de la explosión y la de su estructura temporal. Son
# dos streams separados (ver generate_explosion_sequence): con el mismo índice
# en ambos, dos secuencias distintas compartirían el sorteo temporal.
_TIME_SEED_OFFSET = 10_000_000               # separa el stream temporal del de la explosión


def _block_union(block):
    """Máscara y heatmap del bloque: lo que muestran sus frames juntos.

    Un píxel puede aparecer en más de un frame con clases distintas —una
    trayectoria que pasa por donde antes hubo humo— y gana el más tardío, que es
    el último evento que ocurrió ahí dentro del bloque. Por eso se copian mask y
    heatmap juntos, ceros incluidos: el evento nuevo reemplaza al viejo entero.

    Un píxel cuenta como tocado si tiene mask O heatmap, no solo mask: mask_to_rgb
    escribe la clase trayectoria desde `heatmap > 0`, y hay ~16.700 píxeles por
    secuencia con heatmap sin mask. Filtrando solo por mask, esos quedaban como
    fondo acá y como trayectoria en el target de imagen única — dos etiquetas
    distintas para el mismo píxel según el formato.
    """
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for _, m, h in block:
        tocado = (m > 0) | (h > 0)
        mask[tocado] = m[tocado]
        heatmap[tocado] = h[tocado]
    return mask, heatmap


def generate_single(index: int) -> int:
    """Worker: genera una secuencia, la corta en bloques y los guarda a disco."""
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
        os.makedirs(input_dir, exist_ok=True)
        for i, (tensor, _, _) in enumerate(block):
            tensor_to_image(tensor, os.path.join(input_dir, f"{i:03d}.png"))

        mask, heatmap = _block_union(block)
        mask_to_rgb(mask, heatmap, os.path.join(DATASET_DIR, "targets", f"{name}.png"))

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
