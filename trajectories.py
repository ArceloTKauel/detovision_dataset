"""
trajectories.py - Generación de trayectorias de metralla (rectas y parabólicas).

Dibuja las líneas punteadas que salen de la explosión simulando fragmentos
proyectados. Usa Bresenham para rasterizar las líneas y un sistema de spacing
variable: los puntos son densos cerca del origen y se separan cuadráticamente
con la distancia (ratio² * max_spacing), simulando la desaceleración de la
metralla. Además, cada punto de dibujo inicia una "ráfaga" de 1-5 píxeles
consecutivos (cada uno con 70% de probabilidad) para generar agrupaciones
orgánicas de puntos en vez de puntos solitarios equidistantes. Cada
trayectoria sortea también un ancho (1, 2 o 3 píxeles; ver
_sample_trajectory_width) que se aplica a todos sus puntos.

Todas las funciones aceptan un parámetro opcional mask: si se pasa, marca clase 2
sobre píxeles de fondo (clase 0) en los puntos donde el trazo dejó TINTA, no a lo
largo del recorrido entero. Marcarlo entero dejaba el 85% de la clase sin nada
visible en la entrada. La prioridad de clases en la
máscara es humo > trayectoria > derrumbe > fondo, con una excepción: sobre la
periferia FILAMENTOSA del humo (`filament_region`, ver _SMOKE_OVERRIDE_PROB) la
trayectoria pasa por encima y gana la etiqueta, que es como se ven los "pelos"
de metralla en las referencias reales. Dentro del núcleo de la pluma sigue
oculta.

Funciones:
    - bresenham(y0, x0, y1, x1): Algoritmo de Bresenham para rasterizar una
      línea entre dos puntos. Retorna lista de coordenadas (y, x).
    - draw_trajectory(...): Dibuja una trayectoria recta punteada con ráfagas.
    - draw_returning_parabola(...): Dibuja una trayectoria punteada en forma
      de lazo/óvalo que parte de un centro (dentro de la nube de humo),
      describe una elipse completa (semi-eje mayor = alcance, pudiendo salir
      del lienzo) y cierra exactamente sobre su punto de partida, simulando
      metralla que cae de regreso al punto de impacto.
    - draw_straight_trajectories(...): Genera N trayectorias rectas desde
      centros aleatorios, con longitud mínima = ancho del humo en esa dirección.
    - draw_parabolic_trajectories(...): Genera N trayectorias en lazo
      (draw_returning_parabola) desde centros aleatorios.
    - draw_trajectories(...): Función principal que dibuja ambos tipos.
"""

import numpy as np

from smoke import measure_smoke_width

# Pincel de gradiente para la clase trayectoria en la salida (heatmap):
# kernel gaussiano radial, ancho para que el degradado sea gradual y visible
# (intenso al centro, fino/tenue hacia el borde). Independiente del footprint
# de la máscara categórica (clasificación), que se mantiene angosto.
_HEATMAP_KERNEL_SIZE = 3                     # lado del kernel del gradiente de trayectoria
_HEATMAP_KERNEL_SIGMA = 1.6                  # su sigma

_MASK_OFFSETS = (-1, 0)                      # trazo de 2 px en la máscara categórica


def _make_gradient_kernel(size: int = _HEATMAP_KERNEL_SIZE, sigma: float = _HEATMAP_KERNEL_SIGMA) -> np.ndarray:
    """Kernel gaussiano radial normalizado a pico 1.0 en el centro."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.max()


_HEATMAP_KERNEL = _make_gradient_kernel()

# Clases de mask que tienen prioridad sobre la trayectoria y nunca deben
# recibir gradiente de heatmap (humo=1, derrumbe=3). Prioridad completa:
# humo > trayectoria > derrumbe > fondo.
_HIGHER_PRIORITY_CLASSES = (1, 3)            # humo y derrumbe: la trayectoria no los pisa

# Brillo del tensor de entrada por píxel de trayectoria: en vez de una curva
# determinística por distancia, cada punto dibujado sortea su intensidad de
# una gaussiana truncada al rango [_TRAJECTORY_BRIGHTNESS_RANGE], con la media
# corrida hacia el extremo blanco para que la mayoría de los puntos salgan
# claros pero con variación aleatoria punto a punto. La media se sortea una
# vez por trayectoria (uniforme en _TRAJECTORY_BRIGHTNESS_MEAN_RANGE) para que
# distintas trayectorias tengan distinto nivel de brillo entre sí. Rango
# objetivo de la clase trayectoria (medido con pixel_inspector_gui.py): 2 a 100.
_TRAJECTORY_BRIGHTNESS_RANGE = (2, 100)      # recorte del brillo de un píxel de trazo
_TRAJECTORY_BRIGHTNESS_MEAN_RANGE = (10.0, 80.0)  # media, sorteada una vez por trayectoria
_TRAJECTORY_BRIGHTNESS_STD = 25.0            # desvío de esa gaussiana


def _sample_trajectory_brightness_mean(rng: np.random.Generator) -> float:
    """Sortea la media de brillo de una trayectoria (uniforme, una vez por trayectoria)."""
    return rng.uniform(*_TRAJECTORY_BRIGHTNESS_MEAN_RANGE)


def _trajectory_brightness(rng: np.random.Generator, mean: float) -> int:
    """Sortea el brillo de un píxel de trayectoria de una gaussiana truncada."""
    value = rng.normal(mean, _TRAJECTORY_BRIGHTNESS_STD)
    value = np.clip(value, *_TRAJECTORY_BRIGHTNESS_RANGE)
    return int(value)


# Ancho (en píxeles) de una trayectoria en el tensor de entrada: variable
# categórica sorteada una vez por trayectoria (no por píxel), simulando
# fragmentos de metralla de distinto grosor. La mayoría son de 1 píxel, con
# probabilidad baja de 2 y muy baja de 3. Cada ancho define un bloque de
# offsets cuadrado centrado en el punto dibujado.
_TRAJECTORY_WIDTH_VALUES = (1, 2, 3)         # grosor del trazo, en px
_TRAJECTORY_WIDTH_PROBS = (0.85, 0.12, 0.03)  # casi todas de 1 px
_WIDTH_OFFSETS = {
    1: (0,),
    2: (-1, 0),
    3: (-1, 0, 1),
}


def _sample_trajectory_width(rng: np.random.Generator) -> int:
    """Sortea el ancho de una trayectoria completa (categórica ponderada)."""
    return int(rng.choice(_TRAJECTORY_WIDTH_VALUES, p=_TRAJECTORY_WIDTH_PROBS))


# ── Trayectorias por encima de la periferia filamentosa ────────────────────
# El humo tiene dos partes y la trayectoria se comporta distinto en cada una:
#
#   NÚCLEO (draw_center/draw_smoke/draw_white_blobs) — la trayectoria NACE ahí
#   dentro y queda oculta: atenuada por camouflage_scale y etiquetada humo. Es
#   la lógica de siempre y no cambia. El origen del fragmento no se ve, igual
#   que en las referencias, donde la línea de tiro es una masa saturada sin
#   pelos distinguibles adentro.
#
#   PERIFERIA FIBROSA (draw_smoke_filaments) — la trayectoria pasa POR ENCIMA:
#   se ve y se etiqueta clase 2. En mascara_cambios_final_ESS_F04.png se ve
#   clarísimo: los pelos punteados emergen de la masa central y cruzan el humo
#   fibroso sin interrumpirse hasta salir al fondo negro.
#
# Qué píxel es cuál lo decide smoke.py::draw_smoke_filaments, que devuelve la
# máscara de "humo solo por filamentos"; se propaga hasta acá como
# `filament_region`. Sin esa distinción el pelo tallaría también el núcleo, que
# es el modo de falla de v16 (partir la pluma en fragmentos de trayectoria).
#
# Esto reemplaza al intento de reclasificar estrías de humo al azar (ver nota
# en smoke.py::_FILAMENT_MASK_LEVEL): acá el píxel clase 2 dentro del humo
# SIEMPRE pertenece a una trayectoria que continúa fuera de la pluma, así que
# la regla que aprende el modelo es geométrica y no un sorteo sobre una textura
# idéntica.
#
# Medido en la referencia: el humo fibroso está en p50=42 / p90=61, y los
# puntos de metralla que lo cruzan llegan a 140-177. O sea que el pelo no tiene
# un brillo absoluto propio, sino que se lee POR CONTRASTE sobre el humo local
# — de ahí que el piso sea relativo a tensor[ny, nx] y no un valor fijo.
#
# Dosis del mecanismo: fracción de las trayectorias que cruzan la periferia
# fibrosa y se ven por encima de ella. Queda como constante y no cableado porque
# bajarlo es la forma de aflojarlo sin tocar la geometría.
#
# Bajado de 1.0 a 0.5 el 2026-08-11. Medido sobre 20 semillas, con 1.0 los pelos
# eran el 11.8% del humo que habría sin ellos (rango 7.8-16.0%) — prácticamente
# el mismo tamaño de intervención que el 12% de estrías que reclasificó v19, que
# produjo -46% a -63% de humo en las predicciones sobre imágenes reales. Y v20,
# entrenado con 1.0, da -58.9% de humo en ESS_F04 (la referencia que más ejercita
# el mecanismo) con la razón trayectoria/humo en 1.433 contra 0.532 de v18 —
# dentro del rango de v19. La comparación v18/v20 es limpia: sus pesos de
# entrenamiento son casi iguales (razón trayectoria/fondo 5.7 contra 5.5) y el
# único cambio de dataset entre ambos es este mecanismo.
#
# O sea que el TAMAÑO de la intervención pesa tanto como su coherencia lógica:
# que la regla sea geométrica y no un sorteo sobre textura idéntica (que era la
# justificación para reponer los pelos tras v19) no alcanzó por sí solo.
_SMOKE_OVERRIDE_PROB = 0.5                   # probabilidad de verse sobre la periferia
# Cuánto se levanta el pelo por encima del humo local. Calibrado contra el
# contraste local (píxel menos la mediana de su vecindario 7x7) de la
# estructura fina DENTRO de la pluma en ESS_F04: p90=+10, p99=+26, max=+118.
# Con (35, 75) los pelos salían blanco puro sobre la pluma — el 18.3% por
# encima de 190.
_SMOKE_OVERRIDE_CONTRAST_RANGE = (15, 45)    # cuánto resalta sobre el humo
# Techo absoluto del pelo sobre la pluma: ninguna de las 7 referencias supera
# 190 en ningún píxel, así que un pelo que sature a blanco es un rasgo que el
# modelo no va a ver nunca en producción.
_SMOKE_OVERRIDE_MAX = 190                    # techo de ese realce
# Sobre la periferia el pelo va de 1 px, no del ancho sorteado para el resto de
# la trayectoria: en las referencias los pelos que cruzan la pluma son finos, y
# un trazo de 2-3 px con realce sale como una tira gruesa que no se parece a
# nada real. También es lo que permite que la máscara y la tinta coincidan
# píxel a píxel ahí (ver _paint_traj_mask).
_SMOKE_OVERRIDE_WIDTH = 1                    # grosor del trazo cuando pasa sobre el humo


def _sample_smoke_override(rng: np.random.Generator) -> tuple[bool, int]:
    """Sortea, una vez por trayectoria, si se ve por encima del humo y con
    cuánto contraste sobre el humo local.

    Consume siempre los dos sorteos, aunque el primero salga negativo, para que
    cambiar _SMOKE_OVERRIDE_PROB no desplace el stream del rng ACÁ.

    OJO, esa garantía NO llega hasta el final: el bucle de dibujo de cada
    trayectoria sortea por oportunidad de dibujo (erase, burst, spacing), y
    cuántas veces lo hace depende de si la trayectoria se ve sobre el humo. Desde
    la primera trayectoria afectada el stream se desfasa y las siguientes caen en
    otro lado. Verificado el 2026-08-11 con la misma semilla a dosis 1.0 y 0.0:
    terreno, blast, humo y filamentos salen bit a bit idénticos, pero de los
    ~38.000 píxeles clase 2 solo 3.500-6.300 coinciden entre las dos corridas.

    O sea que NO se puede hacer un A/B por imagen cambiando solo esta constante:
    la diferencia mezcla el efecto del override con otro sorteo de trayectorias.
    Para medir el mecanismo hay que hacerlo sobre UNA corrida — contar los
    píxeles clase 2 que caen dentro de `filament_region`, que es exactamente lo
    que el override convierte de humo en trayectoria."""
    hit = rng.random() < _SMOKE_OVERRIDE_PROB
    contrast = int(rng.uniform(*_SMOKE_OVERRIDE_CONTRAST_RANGE))
    return hit, (contrast if hit else 0)


def _paint_trajectory_pixel(
    tensor: np.ndarray,
    py: int,
    px: int,
    brightness: int,
    width: int,
    mask: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
    override_contrast: int = 0,
    filament_region: np.ndarray | None = None,
) -> None:
    """Pinta un punto de trayectoria como un bloque de `width` x `width`
    píxeles centrado en (py, px), mezclando por máximo y clipeado al lienzo.

    Si se pasa mask, cada píxel del bloque que ya es humo (clase 1) se
    atenúa con camouflage_scale antes de mezclar, para que la trayectoria se
    camufle dentro del humo (oscurecido, ver smoke.py) en vez de sobresalir
    a brillo pleno. Fuera del humo el brillo no se toca.

    override_contrast > 0 invierte ese comportamiento, pero SOLO sobre la
    periferia fibrosa (filament_region): en vez de atenuarse, el píxel se lleva
    a `humo local + override_contrast` cuando eso es más claro que su propio
    brillo. Sin este piso relativo la mezcla por máximo lo borraría, porque el
    brillo sorteado (2-100) casi siempre queda por debajo del humo. Sobre el
    núcleo sigue camuflándose.
    """
    h, w = tensor.shape
    over_filaments = override_contrast > 0 and filament_region is not None
    for dy in _WIDTH_OFFSETS[width]:
        ny = py + dy
        if not (0 <= ny < h):
            continue
        for dx in _WIDTH_OFFSETS[width]:
            nx = px + dx
            if not (0 <= nx < w):
                continue
            pixel_brightness = brightness
            if over_filaments and filament_region[ny, nx]:
                # Se decide por filament_region y NO por mask == 1: otra
                # trayectoria pudo haber pasado antes por acá y dejado el píxel
                # en clase 2, y en ese caso el realce igual tiene que aplicarse.
                pixel_brightness = max(brightness,
                                       min(_SMOKE_OVERRIDE_MAX,
                                           int(tensor[ny, nx]) + override_contrast))
            elif mask is not None and mask[ny, nx] == 1:
                pixel_brightness = int(brightness * camouflage_scale)
            tensor[ny, nx] = max(tensor[ny, nx], pixel_brightness)


def _stamp_heatmap(
    heatmap: np.ndarray,
    py: int,
    px: int,
    mask: np.ndarray | None = None,
    kernel: np.ndarray = _HEATMAP_KERNEL,
) -> None:
    """
    Estampa el kernel de gradiente centrado en (py, px), mezclando por máximo.
    Si se pasa mask, respeta la prioridad de clases: no pinta sobre píxeles
    que ya pertenecen a una clase de mayor prioridad (humo, derrumbe).
    """
    h, w = heatmap.shape
    k = kernel.shape[0]
    half = k // 2
    for ky in range(k):
        ny = py + ky - half
        if not (0 <= ny < h):
            continue
        for kx in range(k):
            nx = px + kx - half
            if not (0 <= nx < w):
                continue
            if mask is not None and mask[ny, nx] in _HIGHER_PRIORITY_CLASSES:
                continue
            value = kernel[ky, kx] * 255
            if value > heatmap[ny, nx]:
                heatmap[ny, nx] = value


def _paint_over_filaments(
    tensor: np.ndarray,
    py: int,
    px: int,
    rng: np.random.Generator,
    brightness_mean: float,
    mask: np.ndarray | None,
    camouflage_scale: float,
    override_contrast: int,
    filament_region: np.ndarray | None,
) -> bool:
    """Sobre la periferia fibrosa el trazo va CONTINUO, no punteado. Pinta el
    píxel y devuelve True si corresponde; False si el píxel no cae ahí y hay que
    seguir con el punteado normal.

    Por qué, y es la razón de ser de todo el mecanismo: la máscara marca clase 2
    en TODO el recorrido, pero el punteado solo deposita tinta en una fracción.
    Medido sobre el código sin esto, el 82% de los píxeles etiquetados
    trayectoria que caen en la periferia fibrosa no recibía nada — o sea que en
    el 82% de su superficie el modelo veía textura de humo fibroso etiquetada
    trayectoria, sin ninguna señal que la distinguiera. Eso es exactamente lo
    que hacía v19, que aprendió que la fibra es metralla y en Video 3 frame 720
    pasó de 8 detecciones a 80.

    La diferencia entre este enfoque y v19 —que la clase 2 esté respaldada por
    tinta visible— solo existe donde efectivamente hay tinta. De ahí el trazo
    continuo.

    Sobre fondo negro el punteado se mantiene: ahí cada punto salta con
    contraste ~51 contra el vacío y el trazo se lee igual.
    """
    if override_contrast <= 0 or filament_region is None:
        return False
    h, w = tensor.shape
    if not (0 <= py < h and 0 <= px < w) or not filament_region[py, px]:
        return False
    brightness = _trajectory_brightness(rng, brightness_mean)
    _paint_trajectory_pixel(tensor, py, px, brightness, _SMOKE_OVERRIDE_WIDTH, mask,
                            camouflage_scale, override_contrast, filament_region)
    return True


def _paint_traj_mask(mask: np.ndarray, py: int, px: int, h: int, w: int,
                     over_smoke: bool = False,
                     filament_region: np.ndarray | None = None) -> None:
    """Marca clase 2 sobre fondo. Con over_smoke, también le gana al humo
    (clase 1) pero solo donde ese humo es periferia fibrosa: sobre el núcleo la
    trayectoria queda oculta, como siempre.

    Sobre la periferia la marca es de 1 px (solo el píxel del recorrido), no de
    2 px como sobre fondo. Es para que la etiqueta coincida con la tinta, que
    ahí también es de 1 px (_SMOKE_OVERRIDE_WIDTH): con la marca de 2 px, la
    mitad de los píxeles clase 2 sobre el humo quedaban sin tinta por
    construcción — el mismo problema que el punteado, en chico.

    El heatmap se estampa DESPUÉS de esto en las tres funciones de dibujo, y
    _stamp_heatmap sigue saltando los píxeles clase 1 — o sea que el gradiente
    cae exactamente sobre los píxeles que acá pasaron a clase 2 y no se
    derrama sobre el humo de al lado. Importa: export.py::mask_to_rgb pinta
    azul TODO píxel con heatmap > 0, así que un derrame convertiría humo en
    trayectoria en el target.
    """
    claim_smoke = over_smoke and filament_region is not None
    for dy in _MASK_OFFSETS:
        for dx in _MASK_OFFSETS:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if mask[ny, nx] == 0:
                mask[ny, nx] = 2
            elif (claim_smoke and mask[ny, nx] == 1 and filament_region[ny, nx]
                    and dy == 0 and dx == 0):
                mask[ny, nx] = 2


# Radio del bloque que marca _stamp_progress alrededor de cada punto del
# recorrido. 1 (bloque 3x3) cubre exactamente todo lo que un punto puede tocar:
# el ancho máximo del trazo (_WIDTH_OFFSETS llega a ±1), el kernel 3x3 del
# heatmap y los offsets de la máscara (_MASK_OFFSETS = -1, 0).
_PROGRESS_RADIUS = 1                         # vecindad que se fecha al anotar el progreso


def _progress_window(launch: float, duration: float) -> tuple[float, float]:
    """Ventana temporal de una trayectoria: en qué momento se lanza y en cuál
    termina de recorrerse, ambos como fracción del tramo post-ignición.

    La duración NO depende del largo del recorrido, y eso está medido, no
    supuesto. Mirando el mismo arco a lo largo de frames consecutivos:

        - Video 4 (f480→f660): un arco de unos 300 px tarda 3 pasos de un tramo
          de ~7.
        - Video 3 (f600→f780): un arco que cruza medio cuadro, muchísimo más
          largo, tarda **los mismos 3 pasos** de un tramo de ~7.

    Los dos dan ~0.43 del tramo con largos que difieren en un orden de magnitud.
    Tiene sentido físico: los fragmentos salen todos en el mismo instante y caen
    con la misma gravedad, así que el tiempo de vuelo es parecido — los que
    llegan más lejos simplemente van más rápido.

    Hacer la duración proporcional al largo (o a su raíz) es lo que producía el
    defecto que esto arregla: los recorridos cortos salían con duración menor a
    un frame y aparecían completos de golpe entre dos frames consecutivos, con
    lazos dando la vuelta entera en un paso. En las referencias eso no pasa
    nunca: todo trazo se dibuja por tramos, avanzando por la punta.

    En 0 la trayectoria ocupa todo el tramo, que es el caso de la generación de
    una sola imagen.

    El lanzamiento se adelanta si con el sorteado la trayectoria no alcanzaría a
    completarse: una trayectoria a medias dejaría fuera un pedazo de recorrido que
    ningún frame llega a mostrar, y entonces la unión de la secuencia ya no
    coincidiría con la imagen que genera el pipeline sin tiempo. Esa equivalencia
    es la propiedad sobre la que se apoya todo sequence.py — en modo acumulado la
    cumple el último frame, y en modo ventana la unión de todos.
    """
    if duration <= 0:
        return (0.0, 1.0)
    duration = min(duration, 1.0)
    launch = min(max(launch, 0.0), 1.0 - duration)
    return (launch, launch + duration)


def _stamp_progress(progress_map: np.ndarray, py: int, px: int, value: float) -> None:
    """Registra en qué momento del recorrido se alcanza cada píxel, quedándose
    con el más temprano.

    Es lo que permite armar la secuencia temporal retrocediendo (ver
    sequence.py): la explosión se dibuja completa UNA vez, igual que siempre, y
    este mapa dice para cada píxel en qué momento aparece. De ahí salen las dos
    lecturas: el frame t es lo que nace en su tramo (modo ventana, el del
    dataset) o todo lo nacido hasta t (modo acumulado).

    Ojo si se usa para el modo ventana: este mapa guarda una sola fecha por
    píxel, la más temprana. Donde dos trayectorias se cruzan, la segunda pasada
    no queda registrada — el píxel aparece en la ventana de la primera. No se
    pierde tinta (la unión de las ventanas cubre exactamente el acumulado, está
    verificado), pero un guion puede quedar con un hueco de pocos píxeles en un
    cruce. No se midió cuánto ocurre en la práctica.

    Que sea el momento del RECORRIDO y no el brillo es el punto: en las
    referencias reales una trayectoria parcial es un trazo truncado —la punta
    avanza y lo ya dibujado no cambia de brillo—, no un trazo atenuado.
    """
    h, w = progress_map.shape
    for ny in range(max(py - _PROGRESS_RADIUS, 0), min(py + _PROGRESS_RADIUS + 1, h)):
        for nx in range(max(px - _PROGRESS_RADIUS, 0), min(px + _PROGRESS_RADIUS + 1, w)):
            if value < progress_map[ny, nx]:
                progress_map[ny, nx] = value


def bresenham(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:
    """Rasterización de línea entre (y0,x0) y (y1,x1). Retorna todos los píxeles."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def draw_trajectory(
    tensor: np.ndarray,
    center: tuple[int, int],
    angle: float,
    length: float,
    origin: tuple[int, int],
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    erase_prob: float = 0.0,
    erase_frac_range: tuple[float, float] = (0.01, 0.05),
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_launch: float = 0.0,
    progress_duration: float = 0.0,
) -> None:
    """
    Dibuja una trayectoria recta punteada desde center en la dirección angle.
    El spacing entre puntos crece cuadráticamente con la distancia al origen:
    cerca = denso, lejos = disperso. Simula desaceleración de metralla.
    erase_prob: probabilidad por oportunidad de dibujo de activar un borrado.
    erase_frac_range: (min, max) fracción del largo total a borrar por sección.
    camouflage_scale: atenuación aplicada a los píxeles que caen sobre humo
    (ver _paint_trajectory_pixel), misma escala global de smoke.py.
    progress_map: si se pasa, se anota ahí en qué punto del recorrido se alcanza
    cada píxel, remapeado al intervalo progress_range de esta trayectoria (ver
    _stamp_progress). No altera el dibujo ni consume rng: la imagen resultante
    es idéntica con y sin él.
    """
    h, w = tensor.shape
    cy, cx = center
    oy, ox = origin

    # Calcular punto final y clipear dentro del lienzo
    end_y = int(round(cy + length * np.sin(angle)))
    end_x = int(round(cx + length * np.cos(angle)))

    end_y = np.clip(end_y, 0, h - 1)
    end_x = np.clip(end_x, 0, w - 1)

    points = bresenham(cy, cx, end_y, end_x)
    total_len = max(len(points), 1)
    brightness_mean = _sample_trajectory_brightness_mean(rng)
    width = _sample_trajectory_width(rng)
    over_smoke, override_contrast = _sample_smoke_override(rng)

    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    inked = []
    max_spacing = 50
    max_dist = length if length > 0 else 1
    prog_lo, prog_hi = _progress_window(progress_launch, progress_duration)

    for idx, (py, px) in enumerate(points):
        if progress_map is not None:
            _stamp_progress(progress_map, py, px,
                            prog_lo + (prog_hi - prog_lo) * (idx / total_len))

        if _paint_over_filaments(tensor, py, px, rng, brightness_mean, mask,
                                  camouflage_scale, override_contrast, filament_region):
            inked.append((py, px))
            continue

        if erase_remaining > 0:
            erase_remaining -= 1
            if erase_remaining == 0:
                grace_remaining = int(rng.uniform(0.05, 0.15) * total_len)
            continue

        if grace_remaining > 0:
            grace_remaining -= 1

        # Ráfaga activa: cada píxel de la ráfaga tiene 70% de probabilidad de dibujarse
        if burst_remaining > 0:
            if rng.random() < 0.7:
                if 0 <= py < h and 0 <= px < w:
                    brightness = _trajectory_brightness(rng, brightness_mean)
                    _paint_trajectory_pixel(tensor, py, px, brightness, width, mask, camouflage_scale,
                                            override_contrast, filament_region)
                    inked.append((py, px))
            burst_remaining -= 1
            continue

        if pixels_since_draw >= next_draw_at:
            if grace_remaining == 0 and rng.random() < erase_prob:
                frac = rng.uniform(erase_frac_range[0], erase_frac_range[1])
                erase_remaining = max(1, int(frac * total_len))
                pixels_since_draw = 0
                next_draw_at = 1
            else:
                if 0 <= py < h and 0 <= px < w:
                    brightness = _trajectory_brightness(rng, brightness_mean)
                    _paint_trajectory_pixel(tensor, py, px, brightness, width, mask, camouflage_scale,
                                            override_contrast, filament_region)
                    inked.append((py, px))

                # Iniciar ráfaga de 1-3 píxeles consecutivos
                burst_remaining = rng.integers(0, 3)

                # Spacing cuadrático: ratio² * max_spacing
                dist_from_origin = np.sqrt((py - oy) ** 2 + (px - ox) ** 2)
                ratio = min(dist_from_origin / max_dist, 1.0)
                spacing = ratio ** 2 * max_spacing
                next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
                pixels_since_draw = 0
        else:
            pixels_since_draw += 1

    # Etiqueta: solo donde cayó tinta, con el footprint de siempre (máscara de
    # 2 px, gaussiano 3x3). Marcarla sobre el recorrido entero dejaba el 85% de la
    # clase sin nada visible, y en un frame de secuencia el 43% de las colas salía
    # 100% vacía.
    if mask is not None:
        for py, px in inked:
            _paint_traj_mask(mask, py, px, h, w, over_smoke, filament_region)

    if heatmap is not None:
        for py, px in inked:
            _stamp_heatmap(heatmap, py, px, mask)


def _ellipse_visible_fraction(
    start: tuple[float, float],
    theta: float,
    a: float,
    b: float,
    side: int,
    h: int,
    w: int,
    num_samples: int = 200,
) -> float:
    """
    Estimación barata (sin rasterizar) de qué fracción del lazo completo cae
    dentro del lienzo [0,h)x[0,w): sortea num_samples puntos a lo largo de la
    elipse paramétrica y mide qué proporción queda dentro de los límites.
    Usada para decidir si una geometría de lazo queda razonablemente contenida
    en el cuadro antes de rasterizarla de verdad.
    """
    sy, sx = start
    uy, ux = np.sin(theta), np.cos(theta)
    vy, vx = side * np.cos(theta), -side * np.sin(theta)
    cy = sy + a * uy
    cx = sx + a * ux

    t = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    py = cy + a * np.cos(t) * uy + b * np.sin(t) * vy
    px = cx + a * np.cos(t) * ux + b * np.sin(t) * vx

    inside = (py >= 0) & (py < h) & (px >= 0) & (px < w)
    return float(inside.mean())


def draw_returning_parabola(
    tensor: np.ndarray,
    start: tuple[float, float],
    origin: tuple[int, int],
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    erase_prob: float = 0.0,
    erase_frac_range: tuple[float, float] = (0.01, 0.05),
    heatmap: np.ndarray | None = None,
    min_visible_fraction: float = 0.0,
    max_attempts: int = 20,
    max_spacing: float = 50.0,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_launch: float = 0.0,
    progress_duration: float = 0.0,
) -> None:
    """
    Dibuja una trayectoria en forma de lazo/óvalo que parte de `start`
    (un centro dentro de la nube de humo), da una vuelta completa en un
    arco amplio y cierra exactamente sobre `start`, simulando metralla que
    describe un arco amplio y cae de regreso al mismo punto de impacto.
    Se rasteriza como elipse completa: semi-eje mayor `a` (alcance),
    semi-eje menor `b` (ancho del lazo).
    Usa Bresenham para interpolar saltos entre pasos consecutivos.
    erase_prob: probabilidad por oportunidad de dibujo de activar un borrado.
    erase_frac_range: (min, max) fracción del largo total a borrar por sección.
    min_visible_fraction: si > 0, resortea la geometría del lazo (hasta
    max_attempts veces) hasta que al menos esa fracción quede dentro del
    lienzo. Con 0.0 (default) no hay restricción: el lazo puede salir
    libremente del cuadro, como hasta ahora.
    max_spacing: separación máxima entre puntos dibujados (en el lanzamiento/
    aterrizaje, donde el spacing es mayor). Valores más chicos dan un
    punteado más cercano/denso en todo el lazo.
    """
    h, w = tensor.shape
    oy, ox = origin
    sy, sx = start
    diagonal = np.hypot(h, w)

    for _ in range(max_attempts):
        theta = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(diagonal * 0.15, diagonal * 0.75)  # semi-eje mayor: alcance del lazo
        b = a * rng.uniform(0.08, 0.35)  # semi-eje menor: ancho del lazo
        side = 1 if rng.random() < 0.5 else -1

        if min_visible_fraction <= 0.0:
            break
        if _ellipse_visible_fraction(start, theta, a, b, side, h, w) >= min_visible_fraction:
            break

    # Eje mayor (u) en dirección theta, eje menor (v) perpendicular
    uy, ux = np.sin(theta), np.cos(theta)
    vy, vx = side * np.cos(theta), -side * np.sin(theta)

    # Centro de la elipse: start queda en el punto t=pi de la parametrización
    cy = sy + a * uy
    cx = sx + a * ux

    sweep = 2 * np.pi  # vuelta completa: el lazo siempre cierra exactamente sobre `start`
    direction = 1 if rng.random() < 0.5 else -1

    num_steps = max(int(a * sweep), 2)
    total_len = max(num_steps, 1)
    brightness_mean = _sample_trajectory_brightness_mean(rng)
    width = _sample_trajectory_width(rng)
    over_smoke, override_contrast = _sample_smoke_override(rng)
    prev_py, prev_px = int(round(sy)), int(round(sx))
    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    inked = []

    max_dist = 2 * a if a > 0 else 1  # distancia del punto más lejano del lazo a `start`
    prog_lo, prog_hi = _progress_window(progress_launch, progress_duration)

    for i in range(num_steps + 1):
        progress = i / num_steps
        t = np.pi + direction * progress * sweep
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        py = int(round(cy + a * cos_t * uy + b * sin_t * vy))
        px = int(round(cx + a * cos_t * ux + b * sin_t * vx))

        if i > 0 and (abs(py - prev_py) > 1 or abs(px - prev_px) > 1):
            segment = bresenham(prev_py, prev_px, py, px)
        else:
            segment = [(py, px)]

        for spy, spx in segment:
            if progress_map is not None:
                _stamp_progress(progress_map, spy, spx,
                                prog_lo + (prog_hi - prog_lo) * progress)

            if _paint_over_filaments(tensor, spy, spx, rng, brightness_mean, mask,
                                      camouflage_scale, override_contrast, filament_region):
                inked.append((spy, spx))
                continue

            if erase_remaining > 0:
                erase_remaining -= 1
                if erase_remaining == 0:
                    grace_remaining = int(rng.uniform(0.05, 0.15) * total_len)
                continue

            if grace_remaining > 0:
                grace_remaining -= 1

            if burst_remaining > 0:
                if rng.random() < 0.7:
                    if 0 <= spy < h and 0 <= spx < w:
                        brightness = _trajectory_brightness(rng, brightness_mean)
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale,
                                                override_contrast, filament_region)
                        inked.append((spy, spx))
                burst_remaining -= 1
                continue

            if pixels_since_draw >= next_draw_at:
                if grace_remaining == 0 and rng.random() < erase_prob:
                    erase_frac = rng.uniform(erase_frac_range[0], erase_frac_range[1])
                    erase_remaining = max(1, int(erase_frac * total_len))
                    pixels_since_draw = 0
                    next_draw_at = 1
                else:
                    if 0 <= spy < h and 0 <= spx < w:
                        brightness = _trajectory_brightness(rng, brightness_mean)
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale,
                                                override_contrast, filament_region)
                        inked.append((spy, spx))

                    burst_remaining = rng.integers(0, 3)

                    # Spacing invertido respecto a la trayectoria recta: acá "lejos
                    # del origen" es el ápice del lazo (máxima altura), donde la
                    # metralla decelera y los puntos deben juntarse; "cerca del
                    # origen" es el lanzamiento/aterrizaje, donde va más rápido.
                    dist_from_origin = np.sqrt((spy - oy) ** 2 + (spx - ox) ** 2)
                    ratio = min(dist_from_origin / max_dist, 1.0)
                    spacing = (1 - ratio) ** 2 * max_spacing
                    next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
                    pixels_since_draw = 0
            else:
                pixels_since_draw += 1

        prev_py, prev_px = py, px

    # Etiqueta: solo donde cayó tinta (ver draw_trajectory).
    if mask is not None:
        for spy, spx in inked:
            _paint_traj_mask(mask, spy, spx, h, w, over_smoke, filament_region)

    if heatmap is not None:
        for spy, spx in inked:
            _stamp_heatmap(heatmap, spy, spx, mask)


def draw_flyover_trajectory(
    tensor: np.ndarray,
    start: tuple[float, float],
    origin: tuple[int, int],
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    erase_prob: float = 0.0,
    erase_frac_range: tuple[float, float] = (0.01, 0.05),
    heatmap: np.ndarray | None = None,
    max_spacing: float = 50.0,
    reach_range: tuple[float, float] = (0.2, 0.45),
    aspect_range: tuple[float, float] = (0.25, 0.55),
    height_factor_range: tuple[float, float] = (1.2, 1.8),
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_launch: float = 0.0,
    progress_duration: float = 0.0,
) -> None:
    """
    Dibuja una trayectoria en forma de arco abierto (medio lazo) que parte de
    `start`, se eleva "sobrevolando" la nube de humo (el ápice queda más
    lejos del origen que el propio borde del humo) y aterriza en otro punto,
    a diferencia de draw_returning_parabola que cierra sobre el mismo punto
    de partida. Simula un fragmento grande que vuela en un arco amplio por
    encima de la explosión.
    Es medio lazo (media elipse, barrido de pi) orientado para que el ápice
    quede siempre del lado contrario al origen (hacia "afuera"/"arriba" de
    la nube), nunca atravesando la explosión.
    reach_range: fracción de min(h, w) usada como semi-eje mayor `a`
    (separación horizontal entre despegue y aterrizaje).
    aspect_range: relación b/a (altura del arco respecto al alcance) para que
    el arco sea achatado tipo "arcoíris" en vez de un semicírculo cerrado.
    height_factor_range: piso mínimo del semi-eje menor `b`, como múltiplo
    del ancho de humo medido desde el origen hacia afuera, para asegurar que
    el ápice quede por encima del borde del humo incluso si `a` sale chico.
    """
    h, w = tensor.shape
    oy, ox = origin
    sy, sx = start

    # Dirección "hacia afuera": del origen hacia el punto de partida. El arco
    # siempre se abre hacia este lado para sobrevolar el humo, nunca hacia
    # el origen.
    out_y, out_x = sy - oy, sx - ox
    out_norm = np.hypot(out_y, out_x)
    if out_norm < 1e-6:
        out_y, out_x, out_norm = -1.0, 0.0, 1.0
    vy, vx = out_y / out_norm, out_x / out_norm

    # Dirección tangencial (perpendicular a la de "afuera"): define hacia
    # dónde queda el punto de aterrizaje respecto al despegue.
    uy, ux = -vx, vy
    if rng.random() < 0.5:
        uy, ux = -uy, -ux

    out_angle = np.arctan2(vy, vx)
    smoke_extent = measure_smoke_width(tensor, origin, out_angle)

    a = rng.uniform(*reach_range) * min(h, w)
    b = max(a * rng.uniform(*aspect_range), smoke_extent * rng.uniform(*height_factor_range))

    cy = sy + a * uy
    cx = sx + a * ux

    apex_y = cy + b * vy
    apex_x = cx + b * vx
    max_dist = max(np.hypot(apex_y - oy, apex_x - ox), 1.0)

    num_steps = max(int(a * np.pi), 2)
    total_len = max(num_steps, 1)
    brightness_mean = _sample_trajectory_brightness_mean(rng)
    width = _sample_trajectory_width(rng)
    over_smoke, override_contrast = _sample_smoke_override(rng)
    prev_py, prev_px = int(round(sy)), int(round(sx))
    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    inked = []
    prog_lo, prog_hi = _progress_window(progress_launch, progress_duration)

    for i in range(num_steps + 1):
        progress = i / num_steps
        t = np.pi - progress * np.pi  # de start (t=pi) a aterrizaje (t=0), pasando por el ápice (t=pi/2)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        py = int(round(cy + a * cos_t * uy + b * sin_t * vy))
        px = int(round(cx + a * cos_t * ux + b * sin_t * vx))

        if i > 0 and (abs(py - prev_py) > 1 or abs(px - prev_px) > 1):
            segment = bresenham(prev_py, prev_px, py, px)
        else:
            segment = [(py, px)]

        for spy, spx in segment:
            if progress_map is not None:
                _stamp_progress(progress_map, spy, spx,
                                prog_lo + (prog_hi - prog_lo) * progress)

            if _paint_over_filaments(tensor, spy, spx, rng, brightness_mean, mask,
                                      camouflage_scale, override_contrast, filament_region):
                inked.append((spy, spx))
                continue

            if erase_remaining > 0:
                erase_remaining -= 1
                if erase_remaining == 0:
                    grace_remaining = int(rng.uniform(0.05, 0.15) * total_len)
                continue

            if grace_remaining > 0:
                grace_remaining -= 1

            if burst_remaining > 0:
                if rng.random() < 0.7:
                    if 0 <= spy < h and 0 <= spx < w:
                        brightness = _trajectory_brightness(rng, brightness_mean)
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale,
                                                override_contrast, filament_region)
                        inked.append((spy, spx))
                burst_remaining -= 1
                continue

            if pixels_since_draw >= next_draw_at:
                if grace_remaining == 0 and rng.random() < erase_prob:
                    erase_frac = rng.uniform(erase_frac_range[0], erase_frac_range[1])
                    erase_remaining = max(1, int(erase_frac * total_len))
                    pixels_since_draw = 0
                    next_draw_at = 1
                else:
                    if 0 <= spy < h and 0 <= spx < w:
                        brightness = _trajectory_brightness(rng, brightness_mean)
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale,
                                                override_contrast, filament_region)
                        inked.append((spy, spx))

                    burst_remaining = rng.integers(0, 3)

                    # Mismo criterio que draw_returning_parabola: lejos del
                    # origen (cerca del ápice) = decelera = puntos densos;
                    # cerca del origen (despegue/aterrizaje) = rápido = disperso.
                    dist_from_origin = np.sqrt((spy - oy) ** 2 + (spx - ox) ** 2)
                    ratio = min(dist_from_origin / max_dist, 1.0)
                    spacing = (1 - ratio) ** 2 * max_spacing
                    next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
                    pixels_since_draw = 0
            else:
                pixels_since_draw += 1

        prev_py, prev_px = py, px

    # Etiqueta: solo donde cayó tinta (ver draw_trajectory).
    if mask is not None:
        for spy, spx in inked:
            _paint_traj_mask(mask, spy, spx, h, w, over_smoke, filament_region)

    if heatmap is not None:
        for spy, spx in inked:
            _stamp_heatmap(heatmap, spy, spx, mask)


def draw_straight_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
) -> None:
    """Genera múltiples trayectorias rectas desde centros aleatorios."""
    h, w = tensor.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)

    for i in range(num_trajectories):
        center = centers[rng.integers(0, len(centers))]
        angle = rng.uniform(0, 2 * np.pi)

        # La longitud mínima es el ancho del humo en esa dirección,
        # para que la trayectoria visible empiece fuera del humo
        smoke_width = measure_smoke_width(tensor, origin, angle)
        min_length = max(10, smoke_width)
        length = rng.uniform(min_length, diagonal)

        if rng.random() < 0.5:
            erase_prob = rng.uniform(0.0, 0.20)
            frac_lo = rng.uniform(0.01, 0.03)
            frac_hi = rng.uniform(frac_lo, 0.05)
        else:
            erase_prob = 0.0
            frac_lo, frac_hi = 0.0, 0.0
        launch, duration = _schedule_at(progress_schedule, i)
        draw_trajectory(tensor, center, angle, length, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                         camouflage_scale, filament_region, progress_map, launch, duration)


# Fracción mínima de trayectorias parabólicas que deben quedar mayormente
# dentro del lienzo por imagen (está bien que varias se salgan del cuadro,
# pero no todas) y umbral de "mayormente contenida" para cada una de ellas.
MIN_CONTAINED_FRACTION = 0.25                # fracción de parábolas forzadas a verse
MIN_VISIBLE_FRACTION = 0.5                   # cuánto de su lazo tiene que entrar en el cuadro

# Rango de max_spacing por trayectoria: valores bajos dan un punteado más
# cercano/denso en todo el lazo; valores altos dan el punteado más disperso
# de antes. Se sortea por trayectoria para tener variedad entre imágenes.
PARABOLA_MAX_SPACING_RANGE = (8, 25)         # separación máxima entre puntos del trazo, en px


def _schedule_at(progress_schedule: list[tuple[float, float]] | None,
                 i: int) -> tuple[float, float]:
    """(lanzamiento, duración) de la i-ésima trayectoria del grupo. Sin
    calendario (generación de una sola imagen) la duración es 0, que
    _progress_window interpreta como "ocupa todo el tramo" — irrelevante de
    todos modos, porque sin progress_map no se anota nada."""
    if progress_schedule is None or i >= len(progress_schedule):
        return (0.0, 0.0)
    return progress_schedule[i]


def draw_parabolic_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
) -> None:
    """
    Genera múltiples trayectorias en forma de lazo/óvalo: cada una parte de
    un centro cercano a la explosión, describe una elipse completa (semi-eje
    mayor y menor aleatorios, pudiendo salir del lienzo) y cierra exactamente
    sobre su punto de partida, simulando metralla que cae de regreso al punto
    de impacto. Al menos MIN_CONTAINED_FRACTION de ellas quedan forzadas a
    tener MIN_VISIBLE_FRACTION de su lazo dentro del cuadro; el resto es
    libre y puede salirse del lienzo sin restricción.
    """
    num_contained = max(1, round(num_trajectories * MIN_CONTAINED_FRACTION))

    for i in range(num_trajectories):
        start = centers[rng.integers(0, len(centers))]

        if rng.random() < 0.5:
            erase_prob = rng.uniform(0.0, 0.20)
            frac_lo = rng.uniform(0.01, 0.03)
            frac_hi = rng.uniform(frac_lo, 0.05)
        else:
            erase_prob = 0.0
            frac_lo, frac_hi = 0.0, 0.0

        min_visible = MIN_VISIBLE_FRACTION if i < num_contained else 0.0
        max_spacing = rng.uniform(*PARABOLA_MAX_SPACING_RANGE)
        launch, duration = _schedule_at(progress_schedule, i)
        draw_returning_parabola(tensor, start, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                                 min_visible, max_spacing=max_spacing, camouflage_scale=camouflage_scale,
                                 filament_region=filament_region, progress_map=progress_map,
                                 progress_launch=launch, progress_duration=duration)


def draw_flyover_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
) -> None:
    """Genera trayectorias de arco abierto que sobrevuelan la nube de humo."""
    for i in range(num_trajectories):
        start = centers[rng.integers(0, len(centers))]

        if rng.random() < 0.5:
            erase_prob = rng.uniform(0.0, 0.20)
            frac_lo = rng.uniform(0.01, 0.03)
            frac_hi = rng.uniform(frac_lo, 0.05)
        else:
            erase_prob = 0.0
            frac_lo, frac_hi = 0.0, 0.0

        max_spacing = rng.uniform(*PARABOLA_MAX_SPACING_RANGE)
        launch, duration = _schedule_at(progress_schedule, i)
        draw_flyover_trajectory(tensor, start, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                                 max_spacing=max_spacing, camouflage_scale=camouflage_scale,
                                 filament_region=filament_region, progress_map=progress_map,
                                 progress_launch=launch, progress_duration=duration)


def draw_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_straight: int,
    num_parabolic: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    num_flyover: int = 0,
    camouflage_scale: float = 1.0,
    filament_region: np.ndarray | None = None,
    progress_map: np.ndarray | None = None,
    progress_schedule: list[tuple[float, float]] | None = None,
) -> None:
    """Punto de entrada: dibuja trayectorias rectas, parabólicas y de sobrevuelo.

    progress_schedule, si se pasa, trae un (lanzamiento, duración) por
    trayectoria en el mismo orden en que se dibujan —primero las rectas, después
    las parabólicas, al final los sobrevuelos— y se reparte entre los tres
    grupos. Lo arma sequence.py; acá solo se distribuye."""
    schedule = progress_schedule or []
    straight_r = schedule[:num_straight] or None
    parabolic_r = schedule[num_straight:num_straight + num_parabolic] or None
    flyover_r = schedule[num_straight + num_parabolic:] or None

    draw_straight_trajectories(tensor, centers, origin, num_straight, rng, mask, heatmap, camouflage_scale,
                                filament_region, progress_map, straight_r)
    draw_parabolic_trajectories(tensor, centers, origin, num_parabolic, rng, mask, heatmap, camouflage_scale,
                                 filament_region, progress_map, parabolic_r)
    draw_flyover_trajectories(tensor, centers, origin, num_flyover, rng, mask, heatmap, camouflage_scale,
                               filament_region, progress_map, flyover_r)
