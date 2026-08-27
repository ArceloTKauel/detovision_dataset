"""
sequence.py - Generación temporal del dataset, por retroceso desde la imagen final.

La explosión se dibuja completa UNA sola vez, anotando para cada píxel a partir
de qué frame existe. Con ese fechado, un frame se arma eligiendo qué rebanada
mostrar, y hay dos lecturas del mismo dato:

    - VENTANA (windowed=True, el modo del dataset): el frame t muestra lo que
      nace en su tramo. Es lo que produce el heatmap de video cuando el
      acumulador se vacía en cada corte, y es lo que hace que apilar frames
      tenga información — con el acumulado el frame t contiene entero al t-1.
    - ACUMULADO (windowed=False): el frame t es la imagen final menos todo lo que
      nace después de t. Su último frame es bit a bit lo que produce
      main.generate_explosion con el mismo rng.

TRAMPA si alguien lo rehace: el "cuándo nace cada píxel" hay que capturarlo
MIENTRAS se dibuja. Deducirlo después desde la imagen final es imposible, porque
la máscara no distingue una trayectoria de otra ni en qué orden se recorrió.

Cada constante de abajo lleva la medición sobre las referencias que la fija.

Uso de la vista previa:
    uv run sequence.py                 explosión al azar
    uv run sequence.py 7               semilla fija, para comparar cambios
    uv run sequence.py 7 36            largo distinto sin tocar la constante
    uv run sequence.py 7 36 acc        el mismo caso en modo acumulado
"""

import os
import sys

import numpy as np
from PIL import Image

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb, contact_sheet
from trajectories import new_progress_map, resolve_progress

# El dominio es oscuro (p50 entre 7 y 10 en las referencias) y sin realce las
# miniaturas del contact sheet se ven negras.
SHEET_BRIGHTNESS = 3.5                       # realce del contact sheet

PREVIEW_DIR = "sequence"                     # frames de la vista previa
PREVIEW_MASK_DIR = "sequence_mask"           # sus máscaras

# Fijo y múltiplo de los 9 frames que entran juntos como canales, así ninguna
# secuencia deja frames colgando sin bloque. Subido de 90 a 180 (2026-08-24):
# con 90, el promedio de trayectoria por frame (total/tramo) no bajaba del
# tope de MAX_TRAJECTORY_PX_PER_FRAME por más que se ajustara la tasa — es
# aritmética, no calibración. Al doble de frames el mismo total se reparte más
# fino, sin tocar main.py ni el balance de clase en imagen única. El precio es
# el dataset: el doble de bloques por secuencia si TOTAL_SEQUENCES no baja a
# la par (ver generate_sequence_dataset.py), y más correlación entre frames de
# una misma explosión si sí baja.
NUM_FRAMES_RANGE = (180, 180)                # largo de la secuencia, múltiplo de BLOCK_SIZE

# En las referencias va de 0.29 a 0.50; acá va más bajo a propósito, porque con
# el valor real el 41% de las máscaras salía 100% fondo. Cuántos frames
# pre-explosión hay es una decisión de MUESTREO nuestra, no una propiedad del
# dominio; lo que sí hay que reproducir es cómo SE VE uno. Siguen haciendo falta:
# son la señal de "acá no hay nada" y atacan el modo de falla de v18.
PRE_IGNITION_FRACTION_RANGE = (0.15, 0.30)   # fracción de frames antes de la ignición

# ── Deriva de cámara ───────────────────────────────────────────────────────
# El terreno no nace ni se intensifica: DERIVA. Un canal del dataset es un
# absdiff, así que no entra el terreno completo sino su RESIDUO — cuánto cambió al
# desplazarse la cámara. Con el terreno completo en los 9 canales, el 95% de un
# canal sintético estaba presente en TODOS los del bloque contra el 4% del real, y
# la regla más fácil de aprender pasaba a ser "lo que no cambia es fondo": el modo
# de falla de v18.
#
# Objetivo del canal real: media 0.8-1.4, p90 2-3, p99 5-7. Con (2.0, 3.0) entran
# la media (0.92) y el p90 (3); el p99 no (14 contra 5-7), porque las curvas de
# nivel sintéticas son bordes duros y desplazar un borde duro devuelve su amplitud
# entera. Atacar la cola pide ablandar terrain.py, no mover esto.
#
# TRAMPA al re-medir: hay que sacar el terreno con un observer sobre la etapa
# "terrain", no llamando a _terrain_blotch_brightness aparte (sobre el campo
# suelto los números dan casi el doble), y con muchas semillas —
# TERRAIN_INTENSITY_RANGE varía el brillo 20x entre imágenes.
TERRAIN_DRIFT_RANGE = (2.0, 3.0)             # deriva de cámara, px por frame

# No es realismo: es lo que diferencia los canales entre sí. El residuo es grande
# donde el desplazamiento cruza una curva de nivel y nulo donde corre paralelo,
# así que con rumbo FIJO se encienden siempre las mismas y los 9 canales quedan
# casi iguales. El precio es que a veces un frame sale casi vacío; bajarlo lo
# empeora, porque daría secuencias enteras flojas en vez de un frame suelto.
TERRAIN_DRIFT_TURN_STD = 0.25                # viraje del rumbo por frame, en radianes

# Al llegar al borde el rumbo REBOTA. Hace falta acotar porque 90 frames a 1-2 px
# son más de 100 px sobre un cuadro de 512, y como los bordes se replican esa
# franja queda con residuo cero. Rebote y no reversión al origen: el tirón le come
# el paso a la cámara (medido, reversión 0.15 da media 0.21 contra 0.57 rebotando).
# Apretar la caja no cuesta nada: paso efectivo 1.37 con 12 px contra 1.40 con 100.
TERRAIN_DRIFT_MAX_OFFSET = 12.0              # caja donde vagabundea la cámara, en px

# Callejón sin salida: la RAMPA del terreno durante la fase pre-explosión. Venía
# de heatmaps ACUMULADOS; en diferencias no aplica, una cámara que deriva a ritmo
# parejo produce residuo constante.

# ── Crecimiento de la pluma ────────────────────────────────────────────────
# Un píxel a distancia r de la línea de tiro nace en (r/alcance)**e del tramo
# post-ignición.
#
# El ALCANCE es un percentil y no el máximo: la masa del humo está mucho más
# adentro que su píxel más lejano (semilla 3: p90 35 contra máximo 69.6), y
# normalizar por el máximo hacía nacer el 90% de la pluma en el primer 36% del
# tramo. Humo y filamentos lo comparten, así el núcleo cierra a media secuencia
# mientras los filamentos siguen alargándose.
#
# El EXPONENTE controla la velocidad de expansión, no el orden de aparición. Con
# 1.0 los fragmentos salen volando cuando la pluma tiene el 15% de su extensión
# final (medido sobre 4 semillas); con 2.0 tiene el 45% y el radio va como la raíz
# del tiempo, que es una expansión desacelerada de gas; con 3.0 aparece de golpe.
SMOKE_REACH_PERCENTILE = 98                  # percentil que define el alcance de la pluma
SMOKE_GROWTH_EXPONENT = 2.0                  # velocidad de expansión: >1 rápido al principio

# Un píxel real no pasa de terreno a humo denso de golpe. Medido sobre ESS_F04 y
# Video 7, la llegada dura p50 3-4 frames (p90 6) y el canal más fuerte se lleva
# la mitad de la tinta. El perfil suma 1.0: reparte la tinta del píxel, no la crea
# — el brillo del humo ya medía bien, el defecto era entregarla toda junta.
#
# NO se aplica a trayectorias ni al fogonazo: de esos dos hay evidencia en contra
# (trazo truncado y no atenuado, fogonazo completo en un solo frame).
SMOKE_ARRIVAL_PROFILE = (0.21, 0.38, 0.24, 0.09, 0.05, 0.02, 0.01)  # reparto de la tinta (suma 1.0)

# Callejón sin salida: dispersar por manchas el momento de nacimiento del humo
# (un Perlin corriendo `frac` hasta ±0.35 del tramo). Indistinguible de no hacer
# nada. Lo que sí resolvió los anillos concéntricos fue la turbulencia.

# Turbulencia: un píxel de humo ya llegado vuelve a encenderse cada tanto — la
# nube real sigue revolviéndose. Target del ACUMULADO del bloque (commit
# f1a2c89, tabla sobre la huella de humo, no el cuadro entero): nuestra máscara
# cubre 1-2.6% del cuadro en un bloque, el real enciende 32-55% de su huella
# por encima de 25.
#
# La FORMA temporal faltaba: medido frame a frame (no por bloque) sobre
# blocks_outputs crudo, más del 60% de los frames reales no tiene ningún píxel
# de señal — la actividad se concentra en ráfagas raras, no en un rumor parejo.
# _FRAME_PROB gatilla si ESTE frame tiene ráfaga en vez de sortear cada píxel
# independiente en los 9 del bloque (daba actividad pareja, nunca un frame en
# cero); _PROB sube para no mover el acumulado: 1-(1-0.377)^3 ≈ 1-(1-0.14)^9.
#
# OJO al recalibrar: las referencias de detovision_segmentation tienen ganancia
# x10 saturada en 255 (subir a 0.60 dio 16-100x más brillante que el real). Las
# crudas están en Desktop/video_diff_heatmap_blocks_outputs.
SMOKE_TURBULENCE_FRAME_PROB = 0.33           # fracción de FRAMES con alguna ráfaga
SMOKE_TURBULENCE_PROB = 0.377                # fracción de nube que se reenciende, en un frame con ráfaga
SMOKE_TURBULENCE_AMPLITUDE = (0.35, 1.0)     # con qué fuerza, en fracción de su valor
SMOKE_TURBULENCE_CELL = 16                   # tamaño de la mancha de ráfaga, en px

# Tope de píxeles de humo NUEVOS por frame, compartido entre llegada y ráfaga
# (alimentan el mismo canal): sin esto, un anillo entero de `smoke_order` nace
# junto y una ráfaga activa reenciende toda la nube ya llegada de una vez. El
# sobrante se difiere a los frames siguientes (`_rate_limit_births`), no se
# pierde. Piso de capacidad: por debajo de total_humo/tramo (~220 con una
# pluma típica de 16 000px y tramo ~75) la nube no terminaría de llegar.
MAX_NEW_SMOKE_PX_PER_FRAME = 250

# El piso no es 0 porque en las referencias las trayectorias aparecen 1-2 imágenes
# DESPUÉS del humo, nunca junto con el fogonazo.
TRAJECTORY_LAUNCH_RANGE = (0.10, 0.70)       # ventana de lanzamiento, en fracción del tramo

# Medido siguiendo el mismo arco por frames consecutivos en dos referencias: ~0.43
# en las dos, pese a que un arco es un orden de magnitud más largo que el otro. Por
# eso el rango es estrecho y NO depende del largo del recorrido.
TRAJECTORY_DURATION_RANGE = (0.30, 0.55)     # tiempo de vuelo, en fracción del tramo

# Se preparan de sobra porque los números reales los sortea generate_explosion con
# su propio rng, después de que acá ya haya que tenerlos listos.
_MAX_TRAJECTORIES = 70                       # cota superior: main.py sortea menos

# Tope de puntos CON TINTA que UNA trayectoria puede estrenar en un frame (no
# compartido entre varias, a diferencia del humo). Es lo que fija el largo del
# segmento que se ve por frame, que en las referencias es ~1px (0.13% del ancho
# medido sobre un bloque real).
#
# La vara es la TINTA y no las posiciones del recorrido: contando posiciones, un
# tramo sólido dejaba entrar 22-35 px con tinta bajo un tope de 35. Contando
# tinta, el tope se cumple sea cual sea la geometría. Ver _stamp_progress_ranked.
#
# Callejón sin salida ya descartado: bajar la ráfaga (burst_remaining). Forzarla
# a 0 deja igual el segmento por frame (p50 36 -> 41).
#
# Aritmética del tope: una trayectoria necesita tinta/tope frames y el tramo son
# ~150, o sea 300 puntos. Con _MAX_DOTTED_INK acotando el punteado, la tinta p50
# es 78 (recta), 94 (sobrevuelo) y 157 (lazo) y el máximo 346, así que solo el
# 2.4% no entra. Esa no se trunca —dejaría tinta que ningún frame muestra— sino
# que COMPRIME el paso y supera el tope; es el precio aceptado.
MAX_TRAJECTORY_PX_PER_FRAME = 2


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


def _rate_limit_births(births: list[np.ndarray], cap: int, last: int) -> list[np.ndarray]:
    """Reparte los nacimientos finitos de una o más capas para que, sumados,
    ningún frame reciba más de `cap` píxeles nuevos: el sobrante se difiere a
    los frames siguientes sin adelantar a nadie ni cambiar el orden natural
    (por eso el sort es estable). Comparten presupuesto porque las capas de la
    lista son la misma clase (smoke + filaments = humo).

    Se acota a `last`: sin esto, una pluma más grande que cap*tramo dejaría
    píxeles con nacimiento después del último frame, y el modo acumulado
    (ver el docstring del módulo) perdería la equivalencia bit a bit con
    generate_explosion en su frame final."""
    values_list, layer_id_list, flat_idx_list = [], [], []
    for li, b in enumerate(births):
        idx = np.flatnonzero(np.isfinite(b))
        if idx.size == 0:
            continue
        values_list.append(b.reshape(-1)[idx])
        layer_id_list.append(np.full(idx.shape, li))
        flat_idx_list.append(idx)
    if not values_list:
        return births

    values = np.concatenate(values_list)
    layer_id = np.concatenate(layer_id_list)
    flat_idx = np.concatenate(flat_idx_list)
    order = np.argsort(values, kind="stable")
    values, layer_id, flat_idx = values[order], layer_id[order], flat_idx[order]

    desired = np.minimum(np.ceil(values), last).astype(np.int64)
    assigned = np.empty(values.size, dtype=np.float32)
    n = values.size
    t, i = desired[0], 0
    while i < n:
        t = min(max(t, desired[i]), last)
        j = min(i + cap, n)
        assigned[i:j] = t
        i = j
        t += 1

    out = [b.copy() for b in births]
    for li in range(len(births)):
        sel = layer_id == li
        out[li].reshape(-1)[flat_idx[sel]] = assigned[sel]
    return out


class _StageRecorder:
    """Observador de generate_explosion: guarda el estado tras cada etapa y, al
    cerrar, fecha los píxeles que cada una agregó.

    El fechado se posterga hasta finalize() porque el alcance de la pluma es
    común a humo y filamentos, y los filamentos se dibujan después.

    Cada capa terminada es (birth, tensor, mask, heatmap), donde birth[y, x] es el
    frame a partir del cual ese píxel muestra el valor de esa capa (inf si la
    etapa no lo tocó). Quedan en orden de dibujo, así que componer un frame es
    aplicarlas en orden quedándose con las ya nacidas.
    """

    def __init__(self, ignition: int, last: int):
        self.ignition = ignition
        self.last = last
        self.terrain: np.ndarray | None = None
        self.stages: list[dict] = []
        self._prev_tensor: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._prev_heatmap: np.ndarray | None = None

    @property
    def _span(self) -> int:
        return max(self.last - self.ignition, 1)

    def __call__(self, stage, tensor, mask, heatmap, ctx) -> None:
        if stage == "terrain":
            # El terreno no se fecha: deriva (ver _terrain_drift). Se guarda
            # aparte porque las manchas sustractivas de draw_smoke lo perforan a
            # negro más adelante, y en los frames pre-explosión ese terreno tiene
            # que verse — desde la imagen final sola sería irrecuperable.
            self.terrain = tensor.copy()
            self._prev_tensor = tensor.copy()
            self._prev_mask = mask.copy()
            self._prev_heatmap = heatmap.copy()
            return

        # El heatmap entra en `changed` aunque solo lo escriba draw_trajectories:
        # hay píxeles donde la trayectoria deja etiqueta sin dejar tinta y sin
        # esto quedarían fuera de su propia capa.
        self.stages.append({
            "name": stage,
            "changed": ((tensor != self._prev_tensor) | (mask != self._prev_mask)
                        | (heatmap != self._prev_heatmap)),
            "tensor": tensor.copy(),
            "mask": mask.copy(),
            "heatmap": heatmap.copy(),
            "ctx": ctx,
        })
        self._prev_tensor = tensor.copy()
        self._prev_mask = mask.copy()
        self._prev_heatmap = heatmap.copy()

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
                # Se fechan por el punto del recorrido en que se alcanzó cada
                # píxel, que trajectories.py anotó al dibujar: es lo que hace que
                # se retraigan por la punta en vez de desvanecerse enteras.
                #
                # La intersección con `changed` importa: progress_map también está
                # definido donde la trayectoria quedó oculta bajo el núcleo, y ahí
                # no dibujó nada. Fecharlos hacía que compose reemitiera el HUMO a
                # brillo pleno — el 15% del recorrido, y la mitad de los píxeles
                # > 100 del canal.
                progress = resolve_progress(s["ctx"]["progress_map"])
                birth = np.where(np.isfinite(progress) & changed,
                                 self.ignition + progress * self._span, np.inf)
            elif s["name"] in ("smoke", "filaments"):
                # El humo se fecha POR CÍRCULO (`smoke_order`, que deja
                # draw_smoke) y no por la distancia del píxel a la línea de tiro.
                # Por distancia funcionaba con el render de zonas, pero con el de
                # aros la tinta vive en círculos dispersos y cortarlos por
                # distancia los rebana en FRANJAS PARALELAS a la línea: la pluma
                # se veía como una barra horizontal de bordes duros.
                #
                # Los filamentos sí se fechan por distancia, que para ellos es
                # correcto: nacen de la línea y se alargan hacia afuera.
                orden = s["ctx"].get("smoke_order")
                base = np.clip(dist / reach, 0.0, 1.0)
                if orden is not None:
                    # los píxeles sin círculo dueño (sub-nubes blancas, que se
                    # dibujan en la misma etapa) caen de vuelta a la distancia
                    base = np.where(np.isfinite(orden), np.nan_to_num(orden), base)
                frac = base ** SMOKE_GROWTH_EXPONENT
                birth = np.where(changed, self.ignition + frac * self._span, np.inf)
            else:
                # Fogonazo y centros: aparecen completos de golpe en la ignición.
                birth = np.where(changed, float(self.ignition), np.inf)

            layers.append((s["name"], birth.astype(np.float32),
                           s["tensor"], s["mask"], s["heatmap"]))

        smoke_idx = [i for i, l in enumerate(layers) if l[0] in ("smoke", "filaments")]
        if smoke_idx:
            capped = _rate_limit_births([layers[i][1] for i in smoke_idx],
                                         MAX_NEW_SMOKE_PX_PER_FRAME, self.last)
            for i, birth in zip(smoke_idx, capped):
                layers[i] = (layers[i][0], birth, *layers[i][2:])
        return layers


def _terrain_drift(num_frames: int, time_rng: np.random.Generator) -> np.ndarray:
    """Posición de la cámara frame a frame, en píxeles: (num_frames + 1, 2). Una
    fila de más porque el frame 0 también es una diferencia y necesita de dónde
    venir. Ver las tres constantes TERRAIN_DRIFT_*.

    El rebote se aplica sobre la posición ya avanzada, doblándola contra la pared:
    el camino recorrido en ese frame sigue midiendo `step`, que es lo que mantiene
    las estadísticas del residuo donde se las calibró."""
    step = time_rng.uniform(*TERRAIN_DRIFT_RANGE)
    heading = time_rng.uniform(0.0, 2 * np.pi)
    turns = time_rng.normal(0.0, TERRAIN_DRIFT_TURN_STD, size=num_frames)
    limit = TERRAIN_DRIFT_MAX_OFFSET

    offsets = np.zeros((num_frames + 1, 2), dtype=np.float32)
    for i, turn in enumerate(turns, start=1):
        heading += turn
        y = offsets[i - 1][0] + step * np.sin(heading)
        x = offsets[i - 1][1] + step * np.cos(heading)
        if abs(y) > limit:
            y = np.copysign(2 * limit, y) - y
            heading = -heading
        if abs(x) > limit:
            x = np.copysign(2 * limit, x) - x
            heading = np.pi - heading
        offsets[i] = (y, x)
    return offsets


def _shift(field: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Traslada un campo (dy, dx) píxeles con interpolación bilineal.

    Los bordes se replican: un wraparound metería un salto de un lado al otro del
    cuadro que en el residuo se vería como una franja encendida. Los índices se
    arman por eje porque una traslación es separable."""
    h, w = field.shape
    yy = np.clip(np.arange(h, dtype=np.float32) - dy, 0, h - 1)
    xx = np.clip(np.arange(w, dtype=np.float32) - dx, 0, w - 1)
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (yy - y0)[:, None]
    fx = (xx - x0)[None, :]
    return (field[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) +
            field[np.ix_(y1, x0)] * fy * (1 - fx) +
            field[np.ix_(y0, x1)] * (1 - fy) * fx +
            field[np.ix_(y1, x1)] * fy * fx)


def _burst_field(height: int, width: int, time_rng: np.random.Generator) -> np.ndarray:
    """Campo [0, 1] de ráfagas de turbulencia para UN frame, por manchas.

    Grilla chica estirada en vez de un Perlin: más barato y alcanza, porque solo
    hace falta que las manchas midan como los remolinos y cambien de frame a
    frame. Ver SMOKE_TURBULENCE_PROB."""
    cell = SMOKE_TURBULENCE_CELL
    small = time_rng.random((max(2, height // cell), max(2, width // cell))).astype(np.float32)
    return np.asarray(Image.fromarray(small).resize((width, height), Image.BILINEAR))


def _terrain_residuals(terrain: np.ndarray, offsets: np.ndarray):
    """Terreno de cada frame: |terreno(t) - terreno(t-1)|, lo mismo que deja un
    absdiff entre dos frames de video.

    Generador y no lista: guardar los 90 serían ~140 MB por worker, y
    generate_sequence_dataset.py corre una docena en paralelo.

    Se redondea en vez de truncar: truncar la interpolación bilineal bajaría la
    media medio nivel de gris, y la media objetivo es 0.8-1.4."""
    previous = _shift(terrain, *offsets[0])
    for offset in offsets[1:]:
        current = _shift(terrain, *offset)
        yield np.rint(np.abs(current - previous)).astype(np.uint8)
        previous = current


def _sample_progress_schedule(time_rng: np.random.Generator) -> list[tuple[float, float]]:
    """(lanzamiento, duración) por trayectoria, en fracción del tramo. Cada
    fragmento sale en su momento y vuela lo suyo: por eso en las referencias unos
    arcos ya están cerrados mientras otros recién empiezan a curvarse."""
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

    windowed=True (el modo del dataset): cada frame muestra SOLO lo que nace en su
    tramo, así que una trayectoria es un guion corto que se desplaza. Ningún frame
    contiene la explosión entera, pero la unión de todos la reconstruye.

    windowed=False: el acumulado. Su último frame es exactamente lo que devuelve
    generate_explosion con el mismo `rng` — la propiedad de validación.

    return_final=True devuelve `(frames, final)`, con `final` = la vista acumulada
    completa.

    `time_rng` va aparte a propósito: sortear la estructura temporal con el rng
    principal correría su stream y rompería esa equivalencia.
    """
    if rng is None:
        rng = np.random.default_rng()
    if time_rng is None:
        time_rng = np.random.default_rng()

    if num_frames is None:
        num_frames = int(time_rng.integers(NUM_FRAMES_RANGE[0], NUM_FRAMES_RANGE[1] + 1))
    ignition = max(1, round(num_frames * time_rng.uniform(*PRE_IGNITION_FRACTION_RANGE)))
    last = num_frames - 1

    progress_map = new_progress_map(height, width)
    recorder = _StageRecorder(ignition, last)
    span = max(last - ignition, 1)

    generate_explosion(
        height, width, rng,
        observer=lambda stage, t, m, hm, ctx: recorder(
            stage, t, m, hm, {**ctx, "progress_map": progress_map}),
        progress_map=progress_map,
        progress_schedule=_sample_progress_schedule(time_rng),
        progress_rate_limit=1.0 / (MAX_TRAJECTORY_PX_PER_FRAME * span),
    )

    # La deriva se sortea siempre, aunque el modo acumulado no la use: así los
    # dos modos consumen `time_rng` igual y la misma semilla da la misma
    # estructura temporal en ambos.
    drift = _terrain_drift(num_frames, time_rng)
    terrain = recorder.terrain.astype(np.float32)
    layers = recorder.finalize()

    def compose(t: int, only_window: bool, terrain_frame: np.ndarray):
        """Arma un frame aplicando las capas fechadas sobre el terreno.

        `only_window` elige cómo se lee el fechado que dejó finalize(): prefijo
        (todo lo nacido hasta t) o banda (solo lo nacido en este tramo).
        `terrain_frame` viene de afuera porque depende del modo (ver la llamada).
        """
        tensor = terrain_frame.astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        heatmap = np.zeros((height, width), dtype=np.uint8)

        for name, birth, layer_tensor, layer_mask, layer_heatmap in layers:
            if not only_window:
                visible = birth <= t
                tensor[visible] = layer_tensor[visible]
                mask[visible] = layer_mask[visible]
                heatmap[visible] = layer_heatmap[visible]
                continue

            # Humo y filamentos LLEGAN: su tinta se reparte entre los frames
            # siguientes al nacimiento (ver SMOKE_ARRIVAL_PROFILE). El resto
            # aparece completo en su frame.
            perfil = SMOKE_ARRIVAL_PROFILE if name in ("smoke", "filaments") else (1.0,)

            # Turbulencia: lo que ya llegó vuelve a encenderse a ráfagas. Va
            # ANTES del bucle de llegada para que, si un píxel está llegando y
            # además le toca ráfaga, gane la llegada — que es el evento nuevo.
            #
            # _FRAME_PROB decide si ESTE frame tiene ráfaga: sin esto, sortear
            # independiente en los 9 frames de un bloque da actividad pareja en
            # los 9 (nunca un frame en cero), y lo real es lo opuesto — ver la
            # nota junto a la constante.
            if name in ("smoke", "filaments"):
                llegado = np.isfinite(birth) & (birth <= t - len(perfil))
                if llegado.any() and time_rng.random() < SMOKE_TURBULENCE_FRAME_PROB:
                    rafaga = _burst_field(height, width, time_rng)
                    sel = llegado & (rafaga > 1.0 - SMOKE_TURBULENCE_PROB)
                    # Mismo tope que la llegada, ver MAX_NEW_SMOKE_PX_PER_FRAME.
                    sel_idx = np.flatnonzero(sel)
                    if sel_idx.size > MAX_NEW_SMOKE_PX_PER_FRAME:
                        keep = time_rng.choice(sel_idx, MAX_NEW_SMOKE_PX_PER_FRAME, replace=False)
                        sel = np.zeros_like(sel)
                        sel.reshape(-1)[keep] = True
                    lo, hi = SMOKE_TURBULENCE_AMPLITUDE
                    # la amplitud sigue al campo: el centro de una ráfaga pega
                    # más fuerte que su borde
                    fuerza = lo + (hi - lo) * (rafaga[sel] - (1.0 - SMOKE_TURBULENCE_PROB)) \
                        / SMOKE_TURBULENCE_PROB
                    tensor[sel] = np.rint(layer_tensor[sel] * fuerza).astype(np.uint8)
                    mask[sel] = layer_mask[sel]
                    heatmap[sel] = layer_heatmap[sel]

            for k, peso in enumerate(perfil):
                visible = (birth > t - k - 1) & (birth <= t - k)
                if not visible.any():
                    continue
                tensor[visible] = np.rint(layer_tensor[visible] * peso).astype(np.uint8)
                mask[visible] = layer_mask[visible]
                heatmap[visible] = layer_heatmap[visible]

        # Mismo saneo que generate_explosion: humo en negro puro no es humo. Se
        # repite por frame porque el recorte temporal puede dejar en negro píxeles
        # que en la imagen final sí tenían tinta.
        mask[(tensor == 0) & (mask == 1)] = 0
        return tensor, mask, heatmap

    # En modo ventana el terreno de cada frame es el residuo de la deriva. En modo
    # acumulado NO: un acumulador que no se vacía vuelve a sumarlo hasta tenerlo
    # entero, y eso es además lo que conserva la equivalencia bit a bit del último
    # frame con generate_explosion.
    if windowed:
        frames = [compose(t, True, terrain_frame) for t, terrain_frame
                  in enumerate(_terrain_residuals(terrain, drift))]
    else:
        frames = [compose(t, False, terrain) for t in range(num_frames)]

    if not return_final:
        return frames

    # Vista final: todo lo que la secuencia repartió entre sus frames. Sirve de
    # target denso para una entrada de N ventanas, que es la única forma de
    # conservar el balance de clases de v20 sin perder la señal temporal.
    return frames, compose(num_frames - 1, False, terrain)


def _reset_preview_dir(path: str) -> None:
    """Deja la carpeta de vista previa vacía de .png antes de escribir: si no, una
    corrida corta deja los frames sobrantes de una larga anterior y el contact
    sheet siguiente mezcla dos explosiones."""
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.endswith(".png"):
            os.remove(os.path.join(path, name))


def main():
    # Argumentos: semilla (repite la misma explosión, para comparar el efecto de
    # un cambio de parámetro), largo de la secuencia y "acc" para el modo
    # acumulado. Ver el docstring del módulo.
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

    # Fuera de la numeración de los frames: es el target denso del bloque, no un
    # frame más de la secuencia.
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
