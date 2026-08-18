"""
main.py - Punto de entrada del generador de dataset sintético de explosiones.

Orquesta el pipeline completo de generación en un solo pase: dibuja el terreno,
crea el lienzo, genera la línea de tiro (la fila de pozos cargados), distribuye
centros de metralla, dibuja el humo y sus filamentos, traza franjas de derrumbe
(opcional) y
trayectorias rectas, parabólicas y de sobrevuelo, y produce simultáneamente dos
salidas:
    - Imagen en escala de grises (entrada del dataset): el humo queda en
      gradiente continuo (más intenso cerca del centro de la explosión, más
      tenue hacia el borde). Las trayectorias también tienen su propio brillo,
      una gaussiana truncada en [2, 100] con la media sorteada por trayectoria
      — no blanco pleno, ver _TRAJECTORY_BRIGHTNESS_RANGE en trajectories.py.
    - Máscara en PNG RGB (salida del dataset): fondo/humo/derrumbe con color
      plano (rojo/verde/amarillo); la trayectoria con gradiente real de azul
      (intenso al centro, tenue hacia el borde), vía el canal heatmap.

Funciones:
    - generate_explosion(height, width, rng): Genera la explosión completa.
      Retorna una tupla (tensor B/W, mask categórica, heatmap de trayectoria).
    - main(): Genera 4 pares de imágenes (explosion_N.png + explosion_N_mask.png).
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

HEIGHT = 512
WIDTH = 768
# Clase 3 (derrumbe) desactivada: implementada y validada, pero fuera del
# alcance del modelo por ahora.
DRAW_LANDSLIDES = False


def generate_explosion(
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
    observer=None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (tensor B/W, mask categórica, heatmap de trayectoria), generados en un solo pase.

    observer, progress_map y progress_schedule son los ganchos de la generación
    temporal (ver sequence.py) y no alteran en nada la imagen resultante: con
    los tres en None esta función produce exactamente lo mismo que antes de que
    existieran, bit a bit y con el mismo consumo de rng.

    observer(stage, tensor, mask, heatmap, ctx) se llama después de cada etapa
    de dibujo, con el estado acumulado y un contexto con la geometría que esa
    etapa necesita para fechar sus píxeles (la línea de tiro, el radio del
    humo). Es lo que permite reconstruir la secuencia hacia atrás: la imagen se
    dibuja UNA vez, completa, y el observador anota de dónde salió cada píxel.
    """
    if rng is None:
        rng = np.random.default_rng()

    def notify(stage: str, **ctx) -> None:
        if observer is not None:
            observer(stage, tensor, mask, heatmap, ctx)

    # Lienzo vacío en escala de grises + máscara de segmentación (0=fondo, 1=humo, 2=trayectoria)
    # + heatmap de gradiente para la clase trayectoria (0 = sin trayectoria, 255 = núcleo)
    tensor = create_canvas(height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    heatmap = np.zeros((height, width), dtype=np.uint8)

    # Manchón de parallax de terreno (ego-motion de cámara): se dibuja PRIMERO,
    # sobre el lienzo todavía vacío, para que humo/trayectoria/derrumbe lo
    # ocluyan de forma natural al dibujarse encima (ver terrain.py). No toca
    # `mask` — el resto de las funciones ya marcan su propia clase de forma
    # incondicional dentro de su geometría, así que el terreno solo queda como
    # fondo donde nada más lo cubre.
    draw_terrain(tensor, rng)
    notify("terrain")

    # Zona de impacto: la fila de pozos cargados (ver canvas.py), con su punto
    # medio como origen. Antes era un cuadrilátero, que daba una pluma
    # redondeada; la carga real es lineal y por eso la pluma sale alargada.
    blast_line = generate_blast_line(height, width, rng)
    mid = blast_line[len(blast_line) // 2]
    origin = (int(round(mid[0])), int(round(mid[1])))

    # Escala de brillo global de esta explosión: compartida entre los centros
    # camuflados y el humo, para que ninguno sature a blanco pleno y el
    # camuflaje siga funcionando (ver smoke.py::sample_brightness_scale).
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
    draw_smoke(tensor, centers, smoke_radius, rng, mask, brightness_scale=brightness_scale)

    # Sub-nubes de "humo blanco": reutiliza draw_smoke sobre un centro y radio
    # más chicos con piso de brillo alto, simulando metralla/brasas
    # incandescentes agrupadas dentro del humo (ver smoke.py::draw_white_blobs).
    # Va ANTES de los filamentos: sortea su centro entre los píxeles ya
    # marcados como humo, así que si los filamentos ya estuvieran marcados
    # podría plantar el blob sobre una estría lejana y fina, desprendido de la
    # nube. Mismo problema que tuvo con el terreno en su momento.
    draw_white_blobs(tensor, smoke_radius, rng, mask)
    notify("smoke", blast_line=blast_line, smoke_radius=smoke_radius)

    # Periferia filamentosa: estrías radiales desde la línea de tiro. Va después
    # de draw_smoke porque sus manchas sustractivas perforan todo lo que esté
    # marcado como humo (ver smoke.py::draw_smoke_filaments).
    # Devuelve qué píxeles son humo SOLO por los filamentos (periferia fibrosa,
    # no núcleo): las trayectorias se dibujan por encima de esos y quedan
    # ocultas dentro del núcleo. Ver trajectories.py::_SMOKE_OVERRIDE_PROB.
    filament_region = draw_smoke_filaments(tensor, blast_line, smoke_radius, rng, mask,
                                            brightness_scale=brightness_scale)
    notify("filaments", blast_line=blast_line, smoke_radius=smoke_radius,
           filament_region=filament_region)

    # Trayectorias de metralla (rectas + parabólicas de ida y vuelta + arcos
    # de sobrevuelo, fragmentos grandes que vuelan por encima de la nube)
    num_straight = rng.integers(15, 30)
    num_parabolic = rng.integers(15, 30)
    num_flyover = rng.integers(1, 4)
    draw_trajectories(tensor, centers, origin, num_straight, num_parabolic, rng, mask, heatmap, num_flyover,
                       camouflage_scale=brightness_scale, filament_region=filament_region,
                       progress_map=progress_map, progress_schedule=progress_schedule)
    notify("trajectories", num_trajectories=num_straight + num_parabolic + num_flyover)

    # Derrumbe: franjas de desprendimiento independientes de la explosión,
    # aproximadamente paralelas entre sí, con puntos de inicio propios
    # distribuidos en cualquier parte del lienzo. Ocurre en ~40% de las imágenes.
    # Se dibuja al final para que respete la prioridad humo > trayectoria >
    # derrumbe > fondo, y nunca pasa por encima de la zona de la explosión
    # (exclusion_radius cubre el centro + los fragmentos + el humo).
    if DRAW_LANDSLIDES and rng.random() < 0.4:
        centers_arr = np.array(centers, dtype=np.float64)
        max_center_dist = np.sqrt(((centers_arr - origin) ** 2).sum(axis=1)).max()
        exclusion_radius = max_center_dist + smoke_radius * 1.3 + 10
        num_stripes = int(np.clip(round(rng.normal(4.5, 1.5)), 1, 8))
        draw_landslides(tensor, num_stripes, np.radians(15), rng, mask, origin, exclusion_radius)

    # Sincronizar mask: píxeles de humo que quedaron en negro puro (p. ej.
    # manchas sustractivas que no dibujaron nada) vuelven a fondo. Las
    # trayectorias (clase 2) no se tocan.
    mask[(tensor == 0) & (mask == 1)] = 0

    return tensor, mask, heatmap


def main():
    for i in range(1, 5):
        tensor, mask, heatmap = generate_explosion(HEIGHT, WIDTH)
        tensor_to_image(tensor, f"explosion_{i}.png")
        mask_to_rgb(mask, heatmap, f"explosion_{i}_mask.png")


if __name__ == "__main__":
    main()
