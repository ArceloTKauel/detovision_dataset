"""
sequence.py - Generación temporal del dataset, por retroceso desde la imagen final.

Las referencias reales no son fotos de instantes sino un MÁXIMO ACUMULADO desde
el frame 0 con ventana creciente (ver video_diff_heatmap_progressive.py en
detovision_segmentation): una imagen cada 60 frames, y nada se apaga nunca. Está
verificado sobre las secuencias reales: entre una imagen y la siguiente, cero
píxeles bajan de valor. La última imagen de una secuencia es exactamente lo que
genera este pipeline hoy.

De ahí la estrategia: en vez de simular la explosión hacia adelante, se dibuja la
imagen final UNA sola vez —el mismo pase de siempre, sin tocar nada— anotando
para cada píxel a partir de qué frame existe, y después se retrocede. El frame t
es la imagen final menos todo lo que nace después de t. Dos consecuencias que
valen el diseño:

    - el último frame es bit a bit lo que produce main.generate_explosion, así
      que la generación temporal no arriesga nada de lo ya validado;
    - la monotonía sale por construcción, no hay que forzarla.

Lo que se midió en las referencias y está codificado acá:

    - 29-50% de cada secuencia es PRE-EXPLOSIÓN, puro terreno (V3 6/15, V4 5/13,
      V7 7/14, V11 4/14). Es la señal que hoy le falta al dataset.
    - El terreno no nace, se INTENSIFICA: las mismas curvas de nivel desde el
      primer frame, ganando contraste hasta saturar cerca de la ignición. Por eso
      lleva una rampa escalar y no un mapa de nacimiento.
    - El fogonazo aparece COMPLETO en un solo frame y fija el máximo de brillo de
      toda la secuencia (V3 197 desde f420, V7 154 desde f480, V11 163 desde f360).
    - El humo crece en radio de forma monótona, rápido al principio.
    - Una trayectoria parcial es un trazo TRUNCADO, no atenuado: la punta avanza
      y lo ya dibujado no cambia de brillo. Los lazos completos solo aparecen en
      los últimos frames porque el fragmento tarda decenas de frames en
      recorrerlos.

Funciones:
    - generate_explosion_sequence(height, width, rng, time_rng): lista de
      (tensor, mask, heatmap), una por frame.
    - main(): escribe una secuencia de ejemplo en el directorio actual
      (sequence_NN.png + sequence_NN_mask.png) más dos contact sheets para
      revisar la progresión completa de un vistazo.
"""

import numpy as np

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb, contact_sheet

# Realce de los contact sheets. El dominio es oscuro (p50 entre 7 y 10 en las
# referencias reales) y sin esto las miniaturas se ven negras.
SHEET_BRIGHTNESS = 3.5

# Largo de la secuencia. Las reales tienen 13-15 imágenes útiles.
NUM_FRAMES_RANGE = (13, 16)

# Fracción de la secuencia anterior a la ignición, medida sobre las cuatro
# referencias miradas en orden: 6/15, 5/13, 7/14, 4/14.
PRE_IGNITION_FRACTION_RANGE = (0.29, 0.50)

# Rampa del terreno: intensidad relativa en el primer frame, y en qué fracción
# de los frames pre-ignición llega a 1.0 y se queda. En las referencias la
# cobertura del terreno crece 3-10x durante la fase pre-explosión y después se
# estabiliza.
TERRAIN_RAMP_START = 0.40
TERRAIN_RAMP_SATURATE_AT = 0.85

# Crecimiento de la pluma. Un píxel a distancia r de la línea de tiro nace en
# (r/alcance)**e del tramo post-ignición.
#
# El alcance NO es la distancia máxima sino un percentil alto: la masa del humo
# está mucho más adentro que su píxel más lejano (medido en la semilla 3: p90=35
# contra un máximo de 69.6), así que normalizar por el máximo hacía nacer el 90%
# de la pluma en el primer 36% del tramo y después no pasaba nada más.
#
# Humo y filamentos comparten el alcance a propósito: los filamentos llegan
# mucho más lejos, y con escalas independientes el núcleo tardaba tanto en
# completarse como la periferia. Con escala común el núcleo cierra a media
# secuencia y los filamentos siguen alargándose hasta el final, que es lo que se
# ve en las referencias.
SMOKE_REACH_PERCENTILE = 98
SMOKE_GROWTH_EXPONENT = 1.0

# Ventana en la que puede lanzarse una trayectoria y cuánto dura su vuelo, ambas
# como fracción del tramo post-ignición. El piso del lanzamiento no es 0 porque
# en las referencias las trayectorias aparecen 1-2 imágenes DESPUÉS del humo,
# nunca junto con el fogonazo.
TRAJECTORY_LAUNCH_RANGE = (0.10, 0.70)
TRAJECTORY_FLIGHT_RANGE = (0.15, 0.55)

# Cota superior de trayectorias por imagen (main.py sortea 15-30 rectas, 15-30
# lazos y 1-3 sobrevuelos). Se preparan rangos de sobra porque los números reales
# los sortea generate_explosion con su propio rng, después de que acá ya haya que
# tenerlos listos.
_MAX_TRAJECTORIES = 70


def _distance_to_blast_line(blast_line: np.ndarray, h: int, w: int) -> np.ndarray:
    """Distancia de cada píxel al punto más cercano de la línea de tiro.

    La pluma no se expande desde un punto sino desde toda la fila de pozos, así
    que el frente de humo es paralelo a la línea, no circular alrededor del
    origen."""
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    dist = np.full((h, w), np.inf, dtype=np.float32)
    for py, px in blast_line:
        np.minimum(dist, np.hypot(yy - py, xx - px), out=dist)
    return dist


class _StageRecorder:
    """Observador de generate_explosion: guarda el estado tras cada etapa y, al
    cerrar, fecha los píxeles que cada una agregó.

    El fechado se posterga hasta finalize() porque el alcance de la pluma es
    común a humo y filamentos, y los filamentos se dibujan después: durante la
    etapa del humo todavía no se sabe hasta dónde va a llegar.

    Cada capa terminada es (birth, tensor, mask, heatmap), donde birth[y, x] es
    el índice de frame a partir del cual ese píxel muestra el valor de esa capa
    (inf si la etapa no lo tocó). Quedan en orden de dibujo, así que componer un
    frame es aplicarlas en orden quedándose con las ya nacidas.
    """

    def __init__(self, ignition: int, last: int):
        self.ignition = ignition
        self.last = last
        self.terrain: np.ndarray | None = None
        self.stages: list[dict] = []
        self._prev_tensor: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None

    @property
    def _span(self) -> int:
        return max(self.last - self.ignition, 1)

    def __call__(self, stage, tensor, mask, heatmap, ctx) -> None:
        if stage == "terrain":
            # El terreno no se fecha: se rampea. Se guarda aparte porque las
            # manchas sustractivas de draw_smoke lo perforan a negro más
            # adelante, y en los frames pre-explosión ese terreno tiene que
            # verse — desde la imagen final sola sería irrecuperable.
            self.terrain = tensor.copy()
            self._prev_tensor = tensor.copy()
            self._prev_mask = mask.copy()
            return

        self.stages.append({
            "name": stage,
            "changed": (tensor != self._prev_tensor) | (mask != self._prev_mask),
            "tensor": tensor.copy(),
            "mask": mask.copy(),
            "heatmap": heatmap.copy(),
            "ctx": ctx,
        })
        self._prev_tensor = tensor.copy()
        self._prev_mask = mask.copy()

    def finalize(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Convierte las etapas grabadas en capas fechadas."""
        h, w = self.terrain.shape
        plume = [s for s in self.stages if s["name"] in ("smoke", "filaments")]

        dist = reach = None
        if plume:
            dist = _distance_to_blast_line(plume[0]["ctx"]["blast_line"], h, w)
            spread = np.concatenate([dist[s["changed"]] for s in plume if s["changed"].any()]
                                    or [np.array([1.0], dtype=np.float32)])
            reach = max(float(np.percentile(spread, SMOKE_REACH_PERCENTILE)), 1e-6)

        layers = []
        for s in self.stages:
            changed = s["changed"]
            if s["name"] == "trajectories":
                # Las trayectorias no se fechan por diferencia sino por el punto
                # del recorrido en que se alcanzó cada píxel, que trajectories.py
                # anotó durante el dibujo. Es lo que hace que la trayectoria se
                # retraiga por la punta en vez de desvanecerse entera.
                progress = s["ctx"]["progress_map"]
                birth = np.where(np.isfinite(progress),
                                 self.ignition + progress * self._span, np.inf)
            elif s["name"] in ("smoke", "filaments"):
                frac = np.clip(dist / reach, 0.0, 1.0) ** SMOKE_GROWTH_EXPONENT
                birth = np.where(changed, self.ignition + frac * self._span, np.inf)
            else:
                # Fogonazo y centros: aparecen completos de golpe en la ignición.
                birth = np.where(changed, float(self.ignition), np.inf)

            layers.append((birth.astype(np.float32), s["tensor"], s["mask"], s["heatmap"]))
        return layers


def _terrain_ramp(num_frames: int, ignition: int) -> np.ndarray:
    """Intensidad relativa del terreno por frame: sube desde TERRAIN_RAMP_START
    y satura antes de la ignición."""
    saturate_at = max(1.0, ignition * TERRAIN_RAMP_SATURATE_AT)
    t = np.minimum(np.arange(num_frames) / saturate_at, 1.0)
    return TERRAIN_RAMP_START + (1.0 - TERRAIN_RAMP_START) * t


def _sample_progress_ranges(time_rng: np.random.Generator) -> list[tuple[float, float]]:
    """(lanzamiento, aterrizaje) por trayectoria, como fracción del tramo
    post-ignición. Cada una se lanza en su propio momento y tarda lo suyo en
    recorrerse: por eso en las referencias los lazos grandes solo están completos
    al final, mientras que los trazos radiales cortos ya se ven temprano."""
    launch = time_rng.uniform(*TRAJECTORY_LAUNCH_RANGE, size=_MAX_TRAJECTORIES)
    flight = time_rng.uniform(*TRAJECTORY_FLIGHT_RANGE, size=_MAX_TRAJECTORIES)
    return [(float(lo), float(min(lo + fl, 1.0))) for lo, fl in zip(launch, flight)]


def generate_explosion_sequence(
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
    time_rng: np.random.Generator | None = None,
    num_frames: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Secuencia acumulada de una explosión: lista de (tensor, mask, heatmap),
    una por frame, del terreno vacío hasta la explosión completa.

    El último elemento es exactamente lo que devuelve generate_explosion con el
    mismo `rng`. `time_rng` es un generador aparte a propósito: sortear la
    estructura temporal con el rng principal correría su stream y cambiaría la
    explosión, rompiendo esa equivalencia.
    """
    if rng is None:
        rng = np.random.default_rng()
    if time_rng is None:
        time_rng = np.random.default_rng()

    if num_frames is None:
        num_frames = int(time_rng.integers(NUM_FRAMES_RANGE[0], NUM_FRAMES_RANGE[1] + 1))
    ignition = max(1, round(num_frames * time_rng.uniform(*PRE_IGNITION_FRACTION_RANGE)))
    last = num_frames - 1

    progress_map = np.full((height, width), np.inf, dtype=np.float32)
    recorder = _StageRecorder(ignition, last)

    generate_explosion(
        height, width, rng,
        observer=lambda stage, t, m, hm, ctx: recorder(
            stage, t, m, hm, {**ctx, "progress_map": progress_map}),
        progress_map=progress_map,
        progress_ranges=_sample_progress_ranges(time_rng),
    )

    ramp = _terrain_ramp(num_frames, ignition)
    terrain = recorder.terrain.astype(np.float32)
    layers = recorder.finalize()

    frames = []
    for t in range(num_frames):
        tensor = (terrain * ramp[t]).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        heatmap = np.zeros((height, width), dtype=np.uint8)

        for birth, layer_tensor, layer_mask, layer_heatmap in layers:
            born = birth <= t
            tensor[born] = layer_tensor[born]
            mask[born] = layer_mask[born]
            heatmap[born] = layer_heatmap[born]

        # Mismo saneo que generate_explosion: humo que quedó en negro puro no es
        # humo. Se repite por frame porque el recorte temporal puede dejar en
        # negro píxeles que en la imagen final sí tenían tinta.
        mask[(tensor == 0) & (mask == 1)] = 0
        frames.append((tensor, mask, heatmap))

    return frames


def main():
    frames = generate_explosion_sequence(HEIGHT, WIDTH)

    tensor_paths, mask_paths = [], []
    for i, (tensor, mask, heatmap) in enumerate(frames):
        tensor_path = f"sequence_{i:02d}.png"
        mask_path = f"sequence_{i:02d}_mask.png"
        tensor_to_image(tensor, tensor_path)
        mask_to_rgb(mask, heatmap, mask_path)
        tensor_paths.append(tensor_path)
        mask_paths.append(mask_path)

    contact_sheet(tensor_paths, "sequence_sheet.png", SHEET_BRIGHTNESS)
    contact_sheet(mask_paths, "sequence_sheet_mask.png")
    print(f"{len(frames)} frames -> sequence_NN.png / sequence_NN_mask.png")
    print("resumen -> sequence_sheet.png / sequence_sheet_mask.png")


if __name__ == "__main__":
    main()
