"""
sequence.py - Generación temporal del dataset, por retroceso desde la imagen final.

La explosión se dibuja completa UNA sola vez —el mismo pase de siempre, sin tocar
nada— anotando para cada píxel a partir de qué frame existe. Con ese fechado por
píxel, un frame se arma eligiendo qué rebanada mostrar, y hay dos lecturas del
mismo dato:

    - VENTANA (windowed=True, el modo del dataset): el frame t muestra solo lo
      que nace en su tramo. Una trayectoria es un guion corto que se desplaza, y
      lo que se dibujó antes ya no está. Es lo que produce el heatmap de video
      cuando el acumulador se vacía en cada corte, y es lo que hace que apilar
      frames tenga información: con el acumulado el frame t contiene entero al
      t-1, así que comparar dos canales no dice nada nuevo.
    - ACUMULADO (windowed=False): el frame t es la imagen final menos todo lo que
      nace después de t. Es la lectura con la que nació este archivo; se conserva
      para comparar y porque su último frame es bit a bit lo que produce
      main.generate_explosion con el mismo rng.

En modo ventana ningún frame contiene la explosión entera, así que esa
equivalencia bit a bit no vale frame a frame — vale para la vista que devuelve
return_final, y está verificado. Lo que sí se comprobó del modo ventana es que la
unión de todos los frames cubre exactamente el acumulado: cero píxeles perdidos y
cero de más.

La trampa si alguien lo rehace: el "cuándo nace cada píxel" hay que capturarlo
MIENTRAS se dibuja. Deducirlo después desde la imagen final y su máscara es
imposible, porque la máscara no distingue una trayectoria de otra ni en qué orden
se recorrió.

Lo que se midió en las referencias y está codificado acá:

    - 29-50% de cada secuencia es PRE-EXPLOSIÓN, puro terreno (V3 6/15, V4 5/13,
      V7 7/14, V11 4/14). Acá va más bajo a propósito: ver
      PRE_IGNITION_FRACTION_RANGE, que explica por qué la proporción de frames
      pre-explosión es una decisión de muestreo nuestra y no una propiedad del
      dominio.
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
      (tensor, mask, heatmap), una por frame. Con return_final=True devuelve
      además la vista acumulada completa, que es el target denso de un bloque.
    - main(): escribe una secuencia de ejemplo en sequence/ y sequence_mask/,
      con el contact sheet y la vista final dentro de cada carpeta.

Uso de la vista previa:
    uv run python sequence.py                 explosión al azar
    uv run python sequence.py 7               semilla fija, para comparar cambios
    uv run python sequence.py 7 36            largo distinto sin tocar la constante
    uv run python sequence.py 7 36 acc        el mismo caso en modo acumulado
"""

import os
import sys

import numpy as np

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb, contact_sheet

# Realce de los contact sheets. El dominio es oscuro (p50 entre 7 y 10 en las
# referencias reales) y sin esto las miniaturas se ven negras.
SHEET_BRIGHTNESS = 3.5

# Carpetas de la vista previa: los frames de una corrida van separados de sus
# máscaras, en vez de intercalados en la raíz del repo. Cada una lleva adentro
# su propio contact sheet.
PREVIEW_DIR = "sequence"
PREVIEW_MASK_DIR = "sequence_mask"

# Largo de la secuencia: fijo, y múltiplo de los 9 frames que entran juntos como
# canales del tensor. 90 son 10 bloques exactos, así que ninguna secuencia deja
# frames colgando sin bloque. Fijo y no sorteado porque un largo variable daría
# bloques incompletos al final.
#
# El precio, medido sobre la semilla 7: repartir la tinta de una explosión entre
# 90 ventanas deja cada frame en 0.107% de trayectoria y 0.090% de humo, y de los
# 10 bloques los tres primeros salen casi sin trayectoria (0.00, 0.00 y 0.01%).
# Con 9 frames el target del único bloque queda en 9.61%. Si hace falta densidad
# por bloque, la palanca no es este número sino la cantidad de trayectorias por
# explosión en main.py.
NUM_FRAMES_RANGE = (90, 90)

# Fracción de la secuencia anterior a la ignición.
#
# En las referencias va de 0.29 a 0.50 (6/15, 5/13, 7/14, 4/14), y ese fue el
# valor inicial. Se bajó a propósito: con 0.29-0.50, el 41% de las máscaras del
# dataset salía 100% fondo y el formato temporal diluía la clase humo a un tercio
# y la trayectoria a un cuarto de su frecuencia en el dataset de imagen única.
#
# La proporción de frames pre-explosión es una decisión de MUESTREO nuestra, no
# una propiedad del dominio: las referencias son grabaciones continuas, no un
# dataset balanceado. Lo que hay que reproducir con fidelidad es cómo se ve un
# frame pre-explosión, no cuántos vienen por secuencia.
#
# Siguen haciendo falta: son la señal de "acá no hay nada" y atacan de frente el
# modo de falla de v18, terreno predicho como humo (ver el repo de segmentación).
PRE_IGNITION_FRACTION_RANGE = (0.15, 0.30)

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

# Ventana en la que puede lanzarse una trayectoria, como fracción del tramo
# post-ignición. El piso no es 0 porque en las referencias las trayectorias
# aparecen 1-2 imágenes DESPUÉS del humo, nunca junto con el fogonazo.
TRAJECTORY_LAUNCH_RANGE = (0.10, 0.70)

# Tiempo de vuelo del fragmento, como fracción del tramo post-ignición.
#
# Medido siguiendo el mismo arco por frames consecutivos en dos referencias:
# Video 4 (f480→f660) y Video 3 (f600→f780) tardan 3 pasos sobre tramos de ~7,
# o sea ~0.43 en las dos — pese a que el arco de Video 3 es un orden de magnitud
# más largo que los de Video 4. Por eso el rango es estrecho y no depende del
# largo del recorrido; el porqué está en trajectories.py::_progress_window.
#
# No depende tampoco de cuántos frames tenga la secuencia: al expresarse como
# fracción del tramo, agregar frames cambia la resolución temporal con que se
# muestrea la explosión, no la física de la explosión.
TRAJECTORY_DURATION_RANGE = (0.30, 0.55)

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


def _sample_progress_schedule(time_rng: np.random.Generator) -> list[tuple[float, float]]:
    """(lanzamiento, duración) por trayectoria, ambos como fracción del tramo
    post-ignición. Cada fragmento sale en su propio momento y vuela lo suyo: por
    eso en las referencias unos arcos ya están cerrados mientras otros recién
    empiezan a curvarse."""
    launch = time_rng.uniform(*TRAJECTORY_LAUNCH_RANGE, size=_MAX_TRAJECTORIES)
    duration = time_rng.uniform(*TRAJECTORY_DURATION_RANGE, size=_MAX_TRAJECTORIES)
    return [(float(lo), float(d)) for lo, d in zip(launch, duration)]


def generate_explosion_sequence(
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
    time_rng: np.random.Generator | None = None,
    num_frames: int | None = None,
    windowed: bool = True,
    return_final: bool = False,
):
    """Secuencia de una explosión: lista de (tensor, mask, heatmap), una por
    frame, del terreno vacío hasta la explosión completa.

    Con `windowed=True` (el modo del dataset) cada frame muestra SOLO lo que nace
    en su tramo: una trayectoria es un guion corto que se desplaza, no un trazo
    que se alarga sobre el anterior. Es lo que produce el heatmap de video cuando
    el acumulador se vacía en cada corte, y es lo que hace que apilar frames
    tenga información — con el acumulado, el frame t contiene entero al t-1 y
    comparar dos canales no dice nada nuevo.

    Con `windowed=False` se recupera el acumulado: cada frame es todo lo nacido
    hasta ese momento. Sirve para comparar y para la propiedad de validación: el
    último elemento es exactamente lo que devuelve generate_explosion con el
    mismo `rng`. En modo ventana esa equivalencia NO vale (ningún frame contiene
    la explosión entera); lo que sigue valiendo es que la unión de todos los
    frames la reconstruye.

    Con `return_final=True` devuelve `(frames, final)` en vez de solo `frames`,
    donde `final` es la vista acumulada completa — la unión de todo lo que la
    secuencia repartió entre sus frames.

    `time_rng` es un generador aparte a propósito: sortear la estructura temporal
    con el rng principal correría su stream y cambiaría la explosión, rompiendo
    esa equivalencia.
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
        progress_schedule=_sample_progress_schedule(time_rng),
    )

    ramp = _terrain_ramp(num_frames, ignition)
    terrain = recorder.terrain.astype(np.float32)
    layers = recorder.finalize()

    def compose(t: int, only_window: bool):
        """Arma un frame aplicando las capas fechadas sobre el terreno.

        `only_window` elige cómo se lee el mismo fechado: prefijo (todo lo nacido
        hasta t) o banda (solo lo nacido en este tramo). El dato de cuándo nace
        cada píxel ya lo dejó finalize().
        """
        tensor = (terrain * ramp[t]).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        heatmap = np.zeros((height, width), dtype=np.uint8)

        for birth, layer_tensor, layer_mask, layer_heatmap in layers:
            visible = ((birth > t - 1) & (birth <= t)) if only_window else (birth <= t)
            tensor[visible] = layer_tensor[visible]
            mask[visible] = layer_mask[visible]
            heatmap[visible] = layer_heatmap[visible]

        # Mismo saneo que generate_explosion: humo que quedó en negro puro no es
        # humo. Se repite por frame porque el recorte temporal puede dejar en
        # negro píxeles que en la imagen final sí tenían tinta.
        mask[(tensor == 0) & (mask == 1)] = 0
        return tensor, mask, heatmap

    frames = [compose(t, windowed) for t in range(num_frames)]
    if not return_final:
        return frames

    # Vista final: el acumulado completo, o sea todo lo que la secuencia llegó a
    # mostrar repartido entre sus frames. Es bit a bit lo que devuelve
    # generate_explosion con el mismo rng, y sirve de target denso para una
    # entrada de N ventanas — que es la única forma de conservar el balance de
    # clases de v20 sin renunciar a la señal temporal en la entrada.
    return frames, compose(num_frames - 1, False)


def _reset_preview_dir(path: str) -> None:
    """Deja la carpeta de vista previa vacía de .png antes de escribir.

    El largo de la secuencia varía entre corridas (NUM_FRAMES_RANGE), así que sin
    esto una corrida corta deja atrás los frames sobrantes de una larga anterior y
    el contact sheet siguiente se compara contra una mezcla de dos explosiones.
    """
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.endswith(".png"):
            os.remove(os.path.join(path, name))


def main():
    # Semilla opcional por línea de comandos: sin ella cada corrida da una
    # explosión distinta, con ella se repite la misma — que es lo que hace falta
    # para comparar el efecto de un cambio de parámetro entre dos corridas.
    #
    # Segundo argumento opcional: largo de la secuencia, para comparar la misma
    # explosión contada en distinta cantidad de frames sin tocar
    # NUM_FRAMES_RANGE. Sin él se sortea como siempre.
    #
    # Tercer argumento "acc": vuelve al render acumulado, para poder mirar la
    # misma explosión en los dos modos sin tocar código.
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    windowed = "acc" not in sys.argv[1:]
    if seed is None:
        frames, final = generate_explosion_sequence(HEIGHT, WIDTH, num_frames=num_frames,
                                                    windowed=windowed, return_final=True)
    else:
        frames, final = generate_explosion_sequence(HEIGHT, WIDTH,
                                                    np.random.default_rng(seed),
                                                    np.random.default_rng(1000 + seed),
                                                    num_frames=num_frames,
                                                    windowed=windowed, return_final=True)
        print(f"semilla {seed}" + (f", {num_frames} frames" if num_frames else "")
              + (", ventana" if windowed else ", acumulado"))

    _reset_preview_dir(PREVIEW_DIR)
    _reset_preview_dir(PREVIEW_MASK_DIR)

    tensor_paths, mask_paths = [], []
    for i, (tensor, mask, heatmap) in enumerate(frames):
        tensor_path = os.path.join(PREVIEW_DIR, f"{i:02d}.png")
        mask_path = os.path.join(PREVIEW_MASK_DIR, f"{i:02d}.png")
        tensor_to_image(tensor, tensor_path)
        mask_to_rgb(mask, heatmap, mask_path)
        tensor_paths.append(tensor_path)
        mask_paths.append(mask_path)

    # Vista final acumulada, fuera de la numeración de los frames: es el target
    # denso del bloque, no un frame más de la secuencia.
    final_tensor, final_mask, final_heatmap = final
    tensor_to_image(final_tensor, os.path.join(PREVIEW_DIR, "final.png"))
    mask_to_rgb(final_mask, final_heatmap, os.path.join(PREVIEW_MASK_DIR, "final.png"))

    contact_sheet(tensor_paths, os.path.join(PREVIEW_DIR, "sheet.png"), SHEET_BRIGHTNESS)
    contact_sheet(mask_paths, os.path.join(PREVIEW_MASK_DIR, "sheet.png"))
    print(f"{len(frames)} frames -> {PREVIEW_DIR}/NN.png / {PREVIEW_MASK_DIR}/NN.png")
    print(f"final    -> {PREVIEW_DIR}/final.png / {PREVIEW_MASK_DIR}/final.png")
    print(f"resumen  -> {PREVIEW_DIR}/sheet.png / {PREVIEW_MASK_DIR}/sheet.png")


if __name__ == "__main__":
    main()
