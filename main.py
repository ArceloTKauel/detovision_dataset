"""
main.py - Pipeline de una explosión, en un solo pase.

Orden de dibujo, que también es el orden de prioridad en la máscara: terreno,
línea de tiro y centros, humo, filamentos, trayectorias y derrumbe. Cada etapa
compone con np.maximum sobre lo anterior, así que lo que se dibuja después ocluye
naturalmente a lo de antes.

Produce las dos salidas del dataset a la vez, compartiendo todos los sorteos:
el tensor en escala de grises (la entrada) y la máscara categórica + heatmap, que
export.py convierte en el PNG RGB (la salida).
"""

import numpy as np

from canvas import (
    create_canvas,
    generate_blast_line,
    draw_center,
    distribute_centers_along_line,
)
from smoke import (
    draw_smoke,
    sample_brightness_scale,
    draw_white_blobs,
    draw_smoke_filaments,
)
from landslide import draw_landslides
from trajectories import draw_trajectories
from terrain import draw_terrain
from export import tensor_to_image, mask_to_rgb

HEIGHT = 512                                 # alto del lienzo, en px
WIDTH = 768                                  # ancho del lienzo, en px
DRAW_LANDSLIDES = False                      # clase 3 desactivada por ahora

# La palanca principal sobre el peso de la clase trayectoria, y los tipos no pesan
# igual: un LAZO deja un recorrido de p50 1080 px contra 264 de una recta, porque
# traza la elipse completa. Con estos valores la clase llega al 15.3% del cuadro
# contra 3.3% del humo, y el 85% de sus píxeles queda bajo 20 en la entrada.
NUM_STRAIGHT_RANGE = (15, 30)                # trayectorias rectas
NUM_PARABOLIC_RANGE = (15, 30)               # lazos que vuelven al origen
NUM_FLYOVER_RANGE = (1, 4)                   # arcos de sobrevuelo


def generate_explosion(
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
    observer=None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
    progress_rate_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (tensor B/W, mask categórica, heatmap de trayectoria).

    observer, progress_map, progress_schedule y progress_rate_limit son los
    ganchos de la generación temporal (ver sequence.py) y no alteran la imagen:
    con los cuatro en None el resultado es idéntico bit a bit y con el mismo
    consumo de rng.

    observer(stage, tensor, mask, heatmap, ctx) se llama después de cada etapa, con
    el estado acumulado y la geometría que esa etapa necesita para fechar sus
    píxeles. Es lo que permite reconstruir la secuencia hacia atrás: la imagen se
    dibuja UNA vez y el observador anota de dónde salió cada píxel.
    """
    if rng is None:
        rng = np.random.default_rng()

    def notify(stage: str, **ctx) -> None:
        if observer is not None:
            observer(stage, tensor, mask, heatmap, ctx)

    # mask: 0=fondo, 1=humo, 2=trayectoria, 3=derrumbe. heatmap: gradiente de la
    # clase trayectoria, 0 = sin trayectoria, 255 = núcleo del trazo.
    tensor = create_canvas(height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    heatmap = np.zeros((height, width), dtype=np.uint8)

    # Terreno primero, sobre el lienzo vacío, para que el resto lo ocluya al
    # dibujarse encima. No toca `mask`: queda como fondo donde nada más lo cubre,
    # que es lo correcto — es un artefacto de cámara, no un objeto a segmentar.
    draw_terrain(tensor, rng)
    notify("terrain")

    # Zona de impacto: la fila de pozos cargados, con su punto medio como origen
    # (ver canvas.py). La carga real es lineal y por eso la pluma sale alargada.
    blast_line = generate_blast_line(height, width, rng)
    mid = blast_line[len(blast_line) // 2]
    origin = (int(round(mid[0])), int(round(mid[1])))

    # Compartida entre los centros camuflados y el humo, para que ninguno sature y
    # el camuflaje siga funcionando (ver smoke.py::sample_brightness_scale).
    brightness_scale = sample_brightness_scale(rng)

    # Centro principal de la explosión
    center_size = rng.integers(1, 3)
    draw_center(tensor, origin, center_size, rng, mask, brightness_scale=brightness_scale)

    # Centros secundarios repartidos a lo largo de la línea (simulan fragmentos)
    num_centers = rng.integers(40, 80)
    centers = distribute_centers_along_line(blast_line, num_centers, rng)

    for center in centers:
        draw_center(tensor, center, rng.integers(0, 2), rng, mask, brightness_scale=brightness_scale)

    # El fogonazo es lo primero que existe: en las referencias aparece completo
    # en un solo frame y fija el máximo de brillo de toda la secuencia.
    notify("blast", blast_line=blast_line, origin=origin)

    # Humo: radio proporcional al tamaño del lienzo
    base_length = min(height, width) * rng.uniform(0.25, 0.40)
    smoke_radius = base_length * rng.uniform(0.15, 0.3)
    # order_map viaja hasta sequence.py por el observer y no altera la imagen.
    smoke_order = np.full((height, width), np.nan, dtype=np.float32)
    draw_smoke(tensor, centers, smoke_radius, rng, mask,
               brightness_scale=brightness_scale, order_map=smoke_order)

    # Sub-nubes incandescentes. Van ANTES de los filamentos: sortean su centro
    # entre los píxeles ya marcados como humo, así que con los filamentos ya
    # marcados podrían plantar un blob sobre una estría lejana, desprendido de la
    # nube — el mismo problema que tuvieron con el terreno en su momento.
    draw_white_blobs(tensor, smoke_radius, rng, mask)
    notify("smoke", blast_line=blast_line, smoke_radius=smoke_radius,
           smoke_order=smoke_order)

    # Periferia filamentosa: estrías radiales desde la línea de tiro. Después de
    # draw_smoke, porque sus manchas sustractivas perforan todo lo marcado como
    # humo. Devuelve qué píxeles son humo SOLO por filamentos: las trayectorias se
    # ven por encima de esos y quedan ocultas dentro del núcleo.
    filament_region = draw_smoke_filaments(tensor, blast_line, smoke_radius, rng, mask,
                                            brightness_scale=brightness_scale)
    notify("filaments", blast_line=blast_line, smoke_radius=smoke_radius,
           filament_region=filament_region)

    num_straight = rng.integers(*NUM_STRAIGHT_RANGE)
    num_parabolic = rng.integers(*NUM_PARABOLIC_RANGE)
    num_flyover = rng.integers(*NUM_FLYOVER_RANGE)
    draw_trajectories(tensor, centers, origin, num_straight, num_parabolic, rng, mask, heatmap, num_flyover,
                       camouflage_scale=brightness_scale, filament_region=filament_region,
                       progress_map=progress_map, progress_schedule=progress_schedule,
                       progress_rate_limit=progress_rate_limit)
    notify("trajectories", num_trajectories=num_straight + num_parabolic + num_flyover)

    # Derrumbe: franjas independientes de la explosión, en ~40% de las imágenes.
    # Al final, para que respete la prioridad de clases, y nunca sobre la zona de
    # la explosión (exclusion_radius cubre centro + fragmentos + humo).
    if DRAW_LANDSLIDES and rng.random() < 0.4:
        centers_arr = np.array(centers, dtype=np.float64)
        max_center_dist = np.sqrt(((centers_arr - origin) ** 2).sum(axis=1)).max()
        exclusion_radius = max_center_dist + smoke_radius * 1.3 + 10
        num_stripes = int(np.clip(round(rng.normal(4.5, 1.5)), 1, 8))
        draw_landslides(tensor, num_stripes, np.radians(15), rng, mask, origin, exclusion_radius)

    # Humo que quedó en negro puro (manchas sustractivas que no dejaron nada)
    # vuelve a fondo. Las trayectorias no se tocan.
    mask[(tensor == 0) & (mask == 1)] = 0

    return tensor, mask, heatmap


def main():
    for i in range(1, 5):
        tensor, mask, heatmap = generate_explosion(HEIGHT, WIDTH)
        tensor_to_image(tensor, f"explosion_{i}.png")
        mask_to_rgb(mask, heatmap, f"explosion_{i}_mask.png")


if __name__ == "__main__":
    main()
