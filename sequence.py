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

Lo que se midió en las referencias y está codificado acá:

    - 29-50% de cada secuencia es pre-explosión, puro terreno. Acá va más bajo a
      propósito: ver PRE_IGNITION_FRACTION_RANGE.
    - El terreno no nace ni se intensifica, DERIVA. Ver TERRAIN_DRIFT_RANGE.
    - El fogonazo aparece completo en un solo frame y fija el máximo de brillo de
      toda la secuencia.
    - El humo crece rápido al principio y además LLEGA a cada píxel repartido en
      3-4 frames. Ver SMOKE_GROWTH_EXPONENT y SMOKE_ARRIVAL_PROFILE.
    - Una trayectoria parcial es un trazo TRUNCADO, no atenuado: la punta avanza
      y lo ya dibujado no cambia de brillo.

Funciones:
    - generate_explosion_sequence(height, width, rng, time_rng): lista de
      (tensor, mask, heatmap), una por frame. Con return_final=True devuelve
      además la vista acumulada completa.
    - main(): escribe una secuencia de ejemplo en sequence/ y sequence_mask/.

Uso de la vista previa:
    uv run python sequence.py                 explosión al azar
    uv run python sequence.py 7               semilla fija, para comparar cambios
    uv run python sequence.py 7 36            largo distinto sin tocar la constante
    uv run python sequence.py 7 36 acc        el mismo caso en modo acumulado
"""

import os
import sys

import numpy as np
from PIL import Image

from main import generate_explosion, HEIGHT, WIDTH
from export import tensor_to_image, mask_to_rgb, contact_sheet

# Realce de los contact sheets. El dominio es oscuro (p50 entre 7 y 10 en las
# referencias reales) y sin esto las miniaturas se ven negras.
SHEET_BRIGHTNESS = 3.5                       # realce del contact sheet: el dominio es oscuro

# Carpetas de la vista previa: los frames de una corrida van separados de sus
# máscaras, en vez de intercalados en la raíz del repo. Cada una lleva adentro
# su propio contact sheet.
PREVIEW_DIR = "sequence"                     # frames de la vista previa
PREVIEW_MASK_DIR = "sequence_mask"           # sus máscaras

# Largo de la secuencia, fijo y múltiplo de los 9 frames que entran juntos como
# canales, así ninguna deja frames colgando sin bloque. Fijo y no sorteado porque
# un largo variable daría bloques incompletos al final.
#
# El precio: repartir la tinta de una explosión entre 90 ventanas deja cada frame
# en ~0.1% de trayectoria y de humo, y los tres primeros bloques salen casi sin
# trayectoria. Si hace falta densidad por bloque, la palanca no es este número
# sino la cantidad de trayectorias por explosión en main.py.
NUM_FRAMES_RANGE = (90, 90)                  # largo de la secuencia, múltiplo de BLOCK_SIZE

# Fracción de la secuencia anterior a la ignición. En las referencias va de 0.29
# a 0.50; acá va más bajo a propósito, porque con el valor real el 41% de las
# máscaras salía 100% fondo.
#
# La proporción de frames pre-explosión es una decisión de MUESTREO nuestra, no
# una propiedad del dominio: las referencias son grabaciones continuas, no un
# dataset balanceado. Lo que hay que reproducir con fidelidad es cómo se ve un
# frame pre-explosión, no cuántos vienen por secuencia. Siguen haciendo falta:
# son la señal de "acá no hay nada" y atacan el modo de falla de v18, terreno
# predicho como humo.
PRE_IGNITION_FRACTION_RANGE = (0.15, 0.30)   # fracción de frames antes de la ignición

# ── Deriva de cámara ───────────────────────────────────────────────────────
# El terreno no nace ni se intensifica: DERIVA. Un canal del dataset representa
# lo mismo que un absdiff entre dos frames reales, así que el terreno no entra
# completo en cada frame sino su RESIDUO — cuánto cambió al desplazarse la cámara.
#
# Importa porque con el terreno completo en los 9 canales, el 95% de un canal
# sintético estaba presente en TODOS los del bloque contra el 4% del real: la
# regla más fácil de aprender pasaba a ser "lo que no cambia es fondo", y en las
# reales el terreno sí cambia. Ese es el modo de falla de v18.
#
# TERRAIN_DRIFT_RANGE, en píxeles por frame, sorteado por secuencia porque los
# videos reales tienen movimientos distintos. Objetivo del canal real: media
# 0.8-1.4, p90 2-3, p99 5-7. Con (2.0, 3.0) la media (0.92) y el p90 (3) entran;
# el p99 no, 14 contra 5-7, porque las curvas de nivel sintéticas son bordes duros
# y desplazar un borde duro devuelve su amplitud entera. Subir la deriva engorda
# la cola más rápido de lo que mueve el centro y bajarla saca la media del rango:
# atacar la cola de verdad pide ablandar el terreno desde terrain.py.
#
# TRAMPA al re-medir: hay que sacar el terreno del pipeline con un observer sobre
# la etapa "terrain", no llamando a _terrain_blotch_brightness aparte — sobre el
# campo suelto sin cuantizar los números dan casi el doble. Y hacen falta muchas
# semillas: TERRAIN_INTENSITY_RANGE varía el brillo 20x entre imágenes.
TERRAIN_DRIFT_RANGE = (2.0, 3.0)             # deriva de cámara, px por frame

# Viraje del rumbo por frame, en radianes. No es realismo: es lo que diferencia
# los canales entre sí. El residuo es grande donde el desplazamiento cruza una
# curva de nivel y casi nulo donde corre paralelo, así que con rumbo FIJO se
# encienden siempre las mismas y los 9 canales quedan casi iguales.
#
# El precio es que de vez en cuando el rumbo queda paralelo a las curvas y ese
# frame sale casi vacío. No es un defecto: una cámara que se mueve a lo largo de
# una curva de nivel no deja residuo. Bajar el viraje lo empeora — en vez de un
# frame flojo cada tanto daría secuencias enteras flojas.
TERRAIN_DRIFT_TURN_STD = 0.25                # viraje del rumbo por frame, en radianes

# Caja dentro de la cual vagabundea la cámara, en píxeles desde el origen; al
# llegar al borde el rumbo REBOTA. Hace falta acotar porque 90 frames a 1-2 px
# son más de 100 px sobre un cuadro de 512, y como los bordes se replican esa
# franja queda con residuo cero y crece.
#
# Rebote y no reversión al origen (un Ornstein-Uhlenbeck como el de landslide.py):
# el tirón le come el paso a la cámara hasta frenarla — medido, reversión 0.15 da
# media 0.21 y p90 0.0 contra 0.57 y 1.0 rebotando. Apretar la caja no cuesta
# nada: el paso efectivo es 1.37 con caja de 12 px contra 1.40 con caja de 100.
TERRAIN_DRIFT_MAX_OFFSET = 12.0              # caja donde vagabundea la cámara, en px

# Callejón sin salida: la RAMPA del terreno, que crecía durante la fase
# pre-explosión. Esa medición venía de heatmaps ACUMULADOS, donde el terreno se
# suma frame a frame. En diferencias no aplica — una cámara que deriva a ritmo
# parejo produce residuo constante, no creciente.

# ── Crecimiento de la pluma ────────────────────────────────────────────────
# Un píxel a distancia r de la línea de tiro nace en (r/alcance)**e del tramo
# post-ignición.
#
# El ALCANCE es un percentil alto y no la distancia máxima: la masa del humo está
# mucho más adentro que su píxel más lejano (semilla 3: p90 35 contra un máximo de
# 69.6), y normalizar por el máximo hacía nacer el 90% de la pluma en el primer
# 36% del tramo. Humo y filamentos comparten alcance para que el núcleo cierre a
# media secuencia mientras los filamentos siguen alargándose.
#
# El EXPONENTE controla la velocidad de expansión, no el orden de aparición. Con
# 1.0 el crecimiento es lineal y los fragmentos salen volando cuando la pluma es
# un hilo: medido sobre 4 semillas, al aparecer la primera trayectoria la pluma
# tenía el 15% de su extensión final. Con 2.0 tiene el 45%, y además el radio va
# como la raíz del tiempo, que es una expansión desacelerada de gas; con 3.0 la
# pluma aparece casi de golpe y se pierde la fase de crecimiento.
SMOKE_REACH_PERCENTILE = 98                  # percentil que define el alcance de la pluma
SMOKE_GROWTH_EXPONENT = 2.0                  # velocidad de expansión: >1 rápido al principio

# Cómo se reparte en el tiempo la LLEGADA del humo a un píxel. Un canal es un
# absdiff, así que muestra cuánto CAMBIÓ el humo en ese frame, y un píxel real no
# pasa de terreno a humo denso de golpe. Medido sobre ESS_F04 y Video 7, aislando
# la racha de canales consecutivos encendidos de cada píxel de evento: la llegada
# dura p50 3-4 frames (p90 6) y el canal más fuerte se lleva la mitad de la tinta.
#
# El perfil suma 1.0: reparte la tinta del píxel, no la crea. Por eso no hubo que
# tocar el brillo del humo, que ya medía bien (94 de mediana contra 82-87 reales);
# el defecto era entregarla toda junta.
#
# NO se aplica a trayectorias ni al fogonazo: de esos dos hay evidencia medida en
# contra (trazo truncado y no atenuado, fogonazo completo en un solo frame).
SMOKE_ARRIVAL_PROFILE = (0.21, 0.38, 0.24, 0.09, 0.05, 0.02, 0.01)  # reparto de la tinta (suma 1.0)

# Callejón sin salida: dispersar por manchas el momento de nacimiento del humo
# (un Perlin corriendo `frac` hasta ±0.35 del tramo), para que la pluma no naciera
# por anillos concéntricos. Medido sobre un bloque, indistinguible de no hacer
# nada. Lo que sí lo resolvió fue la turbulencia.

# Turbulencia: un píxel de humo ya llegado vuelve a encenderse cada tanto, porque
# la nube real no se apaga después de llegar — sigue revolviéndose, y eso es lo que
# hace que EXISTA dentro de un bloque.
#
# El dato que fija el diseño: nuestra máscara de humo cubre 1-2.6% del cuadro en un
# bloque, y el real enciende 32-55% de sus píxeles por encima de 25. No falta nube,
# falta que se encienda. Y se enciende a RÁFAGAS y no con un rumor parejo: la
# actividad media por canal es baja pero el pico es alto, o sea intermitencia, por
# eso _PROB es del orden de 1/9 y la amplitud alta.
#
# OJO al recalibrar: las referencias de detovision_segmentation tienen ganancia x10
# con saturación en 255. Las crudas están en Desktop/video_diff_heatmap_blocks_outputs
# y son las que valen — un bloque real da >10 0.13%, >25 0.012%, p99 5, y contra las
# amplificadas se concluye lo contrario. Subir esta constante a 0.60 contra la fuente
# amplificada nos dejó 16-100x más brillantes que el dominio real.
#
# El campo de ráfagas es POR MANCHAS y se resortea en cada frame: fijo estaría
# igual en los 9 canales y volvería el problema del terreno estático.
SMOKE_TURBULENCE_PROB = 0.14                 # fracción de nube que se reenciende por frame
SMOKE_TURBULENCE_AMPLITUDE = (0.35, 1.0)     # con qué fuerza, en fracción de su valor
SMOKE_TURBULENCE_CELL = 16                   # tamaño de la mancha de ráfaga, en px

# Ventana en la que puede lanzarse una trayectoria, como fracción del tramo
# post-ignición. El piso no es 0 porque en las referencias las trayectorias
# aparecen 1-2 imágenes DESPUÉS del humo, nunca junto con el fogonazo.
TRAJECTORY_LAUNCH_RANGE = (0.10, 0.70)       # ventana de lanzamiento, en fracción del tramo

# Tiempo de vuelo del fragmento, como fracción del tramo post-ignición. Medido
# siguiendo el mismo arco por frames consecutivos en dos referencias: ~0.43 en las
# dos, pese a que un arco es un orden de magnitud más largo que el otro. Por eso
# el rango es estrecho y NO depende del largo del recorrido ni de cuántos frames
# tenga la secuencia.
TRAJECTORY_DURATION_RANGE = (0.30, 0.55)     # tiempo de vuelo, en fracción del tramo

# Cota superior de trayectorias por imagen (main.py sortea 15-30 rectas, 15-30
# lazos y 1-3 sobrevuelos). Se preparan rangos de sobra porque los números reales
# los sortea generate_explosion con su propio rng, después de que acá ya haya que
# tenerlos listos.
_MAX_TRAJECTORIES = 70                       # cota superior: main.py sortea menos


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
        # hay píxeles donde la trayectoria deja etiqueta sin dejar tinta (el 73%
        # de su recorrido, ver finalize) y sin esto quedarían fuera de su propia
        # capa.
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
                # Las trayectorias no se fechan por diferencia sino por el punto
                # del recorrido en que se alcanzó cada píxel, que trajectories.py
                # anotó al dibujar. Es lo que hace que se retraigan por la punta
                # en vez de desvanecerse enteras.
                #
                # Se INTERSECA con `changed`, y ese detalle importa: el
                # progress_map está definido también donde la trayectoria quedó
                # oculta bajo el núcleo del humo, y ahí no dibujó nada. Fecharlos
                # hacía que compose reemitiera el HUMO a brillo pleno en el frame
                # en que pasaba la trayectoria — el 15% del recorrido, y la mitad
                # de los píxeles > 100 del canal.
                progress = s["ctx"]["progress_map"]
                birth = np.where(np.isfinite(progress) & changed,
                                 self.ignition + progress * self._span, np.inf)
            elif s["name"] in ("smoke", "filaments"):
                # El humo se fecha POR CÍRCULO, no por la distancia del píxel a
                # la línea de tiro: draw_smoke deja en `smoke_order` el orden de
                # nacimiento del círculo que posee cada píxel.
                #
                # Fechar por distancia del píxel funcionaba con el render de
                # zonas, donde la pluma era un campo de distancia suave. Con el
                # render de aros la tinta vive en círculos dispersos, y cortarlos
                # por distancia los rebana en FRANJAS PARALELAS a la línea de
                # tiro: medido, todos los píxeles que nacían en un mismo frame
                # quedaban a la misma distancia con desviación de 1-2 px, y la
                # pluma se veía como una barra horizontal de bordes duros.
                #
                # Los filamentos siguen fechándose por distancia, que para ellos
                # es correcto: nacen de la línea y se alargan hacia afuera.
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
        return layers


def _terrain_drift(num_frames: int, time_rng: np.random.Generator) -> np.ndarray:
    """Posición de la cámara frame a frame, en píxeles: (num_frames + 1, 2).

    Una fila de más porque el frame 0 también es una diferencia y necesita de
    dónde venir; la primera es el origen.

    El paso se sortea una vez por secuencia (los videos reales tienen
    movimientos de cámara distintos), el rumbo vira frame a frame y rebota
    contra las paredes de la caja — ver las tres constantes TERRAIN_DRIFT_*.

    El rebote se aplica sobre la posición ya avanzada, doblándola contra la
    pared: el camino recorrido en ese frame sigue midiendo `step`, que es lo que
    mantiene las estadísticas del residuo donde se las calibró."""
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

    Los bordes se replican: un wraparound metería un salto artificial de un lado
    al otro del cuadro que en el residuo se vería como una franja encendida.

    Los índices se arman por eje y se combinan con np.ix_ porque una traslación
    es separable — no hace falta una grilla completa de coordenadas."""
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

    Se arma sorteando una grilla chica y estirándola: es mucho más barato que un
    Perlin y alcanza, porque acá solo hace falta que las manchas tengan un tamaño
    parecido al de los remolinos y que cambien de frame a frame. Se sortea una
    por frame a propósito — ver SMOKE_TURBULENCE_PROB."""
    cell = SMOKE_TURBULENCE_CELL
    small = time_rng.random((max(2, height // cell), max(2, width // cell))).astype(np.float32)
    return np.asarray(Image.fromarray(small).resize((width, height), Image.BILINEAR))


def _terrain_residuals(terrain: np.ndarray, offsets: np.ndarray):
    """Terreno de cada frame: |terreno(t) - terreno(t-1)|, o sea lo que cambió al
    moverse la cámara — lo mismo que deja un absdiff entre dos frames de video.

    Es un generador y no una lista porque el terreno de un frame pesa lo mismo
    que el cuadro entero: guardar los 90 serían ~140 MB por worker, y
    generate_sequence_dataset.py corre una docena en paralelo.

    Se redondea en vez de truncar: los canales reales son un absdiff de enteros,
    y truncar la interpolación bilineal bajaría la media medio nivel de gris —
    sobre una media objetivo de 0.8-1.4 eso no es un detalle."""
    previous = _shift(terrain, *offsets[0])
    for offset in offsets[1:]:
        current = _shift(terrain, *offset)
        yield np.rint(np.abs(current - previous)).astype(np.uint8)
        previous = current


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

    # La deriva se sortea siempre, aunque el modo acumulado no la use: así los
    # dos modos consumen `time_rng` igual y la misma semilla da la misma
    # estructura temporal en ambos.
    drift = _terrain_drift(num_frames, time_rng)
    terrain = recorder.terrain.astype(np.float32)
    layers = recorder.finalize()

    def compose(t: int, only_window: bool, terrain_frame: np.ndarray):
        """Arma un frame aplicando las capas fechadas sobre el terreno.

        `only_window` elige cómo se lee el mismo fechado: prefijo (todo lo nacido
        hasta t) o banda (solo lo nacido en este tramo). El dato de cuándo nace
        cada píxel ya lo dejó finalize().

        `terrain_frame` es el terreno de ESE frame, que depende del modo y por
        eso viene de afuera: el residuo de la deriva en modo ventana, el terreno
        entero en modo acumulado (ver la llamada).
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
            # aparece completo en su frame, que es lo que muestran las
            # referencias para el trazo de trayectoria y para el fogonazo.
            perfil = SMOKE_ARRIVAL_PROFILE if name in ("smoke", "filaments") else (1.0,)

            # Turbulencia: lo que ya llegó vuelve a encenderse a ráfagas. Va
            # ANTES del bucle de llegada para que, si un píxel está llegando y
            # además le toca ráfaga, gane la llegada — que es el evento nuevo.
            if name in ("smoke", "filaments"):
                llegado = np.isfinite(birth) & (birth <= t - len(perfil))
                if llegado.any():
                    rafaga = _burst_field(height, width, time_rng)
                    sel = llegado & (rafaga > 1.0 - SMOKE_TURBULENCE_PROB)
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

        # Mismo saneo que generate_explosion: humo que quedó en negro puro no es
        # humo. Se repite por frame porque el recorte temporal puede dejar en
        # negro píxeles que en la imagen final sí tenían tinta.
        mask[(tensor == 0) & (mask == 1)] = 0
        return tensor, mask, heatmap

    # En modo ventana el terreno de cada frame es el residuo de la deriva, que es
    # lo que deja un absdiff. En modo acumulado NO: un acumulador que no se vacía
    # vuelve a sumar el terreno hasta tenerlo entero, así que ahí entra completo
    # desde el primer frame — que además es lo que conserva la equivalencia bit a
    # bit del último frame con generate_explosion.
    if windowed:
        frames = [compose(t, True, terrain_frame) for t, terrain_frame
                  in enumerate(_terrain_residuals(terrain, drift))]
    else:
        frames = [compose(t, False, terrain) for t in range(num_frames)]

    if not return_final:
        return frames

    # Vista final: el acumulado completo, o sea todo lo que la secuencia llegó a
    # mostrar repartido entre sus frames. Es bit a bit lo que devuelve
    # generate_explosion con el mismo rng, y sirve de target denso para una
    # entrada de N ventanas — que es la única forma de conservar el balance de
    # clases de v20 sin renunciar a la señal temporal en la entrada.
    return frames, compose(num_frames - 1, False, terrain)


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
