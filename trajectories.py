"""
trajectories.py - Trayectorias de metralla: rectas, lazos y arcos de sobrevuelo.

Cada trayectoria se rasteriza con Bresenham y se dibuja PUNTEADA: el spacing
crece cuadráticamente con la distancia al origen (la metralla desacelera) y cada
punto abre una ráfaga de 1-5 píxeles, para que el trazo sean agrupaciones y no
puntos equidistantes. El ancho y el brillo medio se sortean una vez por
trayectoria.

La etiqueta (clase 2) se marca SOLO donde el punteado dejó tinta, no sobre el
recorrido entero: marcarlo entero dejaba el 85% de la clase sin nada visible en
la entrada.

Prioridad de clases: humo > trayectoria > derrumbe > fondo. La excepción es la
periferia FILAMENTOSA del humo, donde la trayectoria pasa por encima y gana la
etiqueta —así se ven los "pelos" de metralla en las referencias—; dentro del
núcleo de la pluma sigue oculta. Ver _SMOKE_OVERRIDE_PROB.
"""

import numpy as np

from smoke import measure_smoke_width

# Pincel del gradiente de trayectoria en la salida (heatmap): gaussiana radial,
# independiente del footprint de la máscara categórica, que va más angosto.
_HEATMAP_KERNEL_SIZE = 3                     # lado del kernel del gradiente
_HEATMAP_KERNEL_SIGMA = 1.6                  # su sigma

_MASK_OFFSETS = (-1, 0)                      # trazo de 2 px en la máscara categórica


def _make_gradient_kernel(size: int = _HEATMAP_KERNEL_SIZE, sigma: float = _HEATMAP_KERNEL_SIGMA) -> np.ndarray:
    """Kernel gaussiano radial normalizado a pico 1.0 en el centro."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.max()


_HEATMAP_KERNEL = _make_gradient_kernel()

_HIGHER_PRIORITY_CLASSES = (1, 3)            # humo y derrumbe: la trayectoria no los pisa

# Brillo de un píxel de trazo: gaussiana truncada, con la media sorteada una vez
# por trayectoria para que unas salgan más claras que otras. Rango objetivo de la
# clase, medido con pixel_inspector_gui.py: 2 a 100.
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


# Ancho del trazo, sorteado una vez por trayectoria y no por píxel: fragmentos de
# distinto grosor. Cada valor define un bloque de offsets centrado en el punto.
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
# El humo tiene dos partes y la trayectoria se comporta distinto en cada una: en
# el NÚCLEO nace adentro y queda oculta (atenuada por camouflage_scale, etiquetada
# humo); sobre la PERIFERIA FIBROSA pasa por encima, se ve y se etiqueta clase 2,
# que es como se ven los "pelos" de metralla en mascara_cambios_final_ESS_F04.png.
# Qué píxel es cuál lo decide smoke.py::draw_smoke_filaments y llega hasta acá
# como `filament_region`; sin esa distinción el pelo tallaría también el núcleo y
# partiría la pluma en fragmentos de trayectoria.
#
# Dosis: a 1.0 los pelos eran el 11.8% del humo y v20 perdió 59% de humo en
# ESS_F04; 0.5 es media dosis del mismo mecanismo. Es la palanca para aflojarlo
# sin tocar la geometría.
_SMOKE_OVERRIDE_PROB = 0.5                   # probabilidad de verse sobre la periferia
# El pelo no tiene brillo propio: se lee POR CONTRASTE sobre el humo local, por
# eso el piso es relativo a tensor[ny, nx]. Medido en ESS_F04, el contraste local
# de la estructura fina dentro de la pluma es p90 +10 / p99 +26.
_SMOKE_OVERRIDE_CONTRAST_RANGE = (15, 45)    # cuánto resalta sobre el humo
_SMOKE_OVERRIDE_MAX = 190                    # techo: ninguna de las 7 referencias lo pasa
_SMOKE_OVERRIDE_WIDTH = 1                    # 1 px: los pelos reales son finos


def _sample_smoke_override(rng: np.random.Generator) -> tuple[bool, int]:
    """Sortea, una vez por trayectoria, si se ve por encima del humo y con cuánto
    contraste. Consume los dos sorteos siempre, aunque el primero salga negativo.

    Esa garantía no llega hasta el final: el bucle de dibujo sortea distinta
    cantidad de veces según si la trayectoria se ve o no. O sea que NO se puede
    hacer un A/B cambiando solo _SMOKE_OVERRIDE_PROB — verificado, con la misma
    semilla a dosis 1.0 y 0.0 solo coinciden 3.500-6.300 de ~38.000 píxeles clase
    2. Para medir el mecanismo hay que contar, sobre UNA corrida, los píxeles
    clase 2 que caen dentro de `filament_region`."""
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
    """Pinta un punto como un bloque de `width` x `width` centrado en (py, px),
    mezclando por máximo y clipeado al lienzo.

    Con mask, un píxel que ya es humo se atenúa con camouflage_scale para que la
    trayectoria se camufle dentro de la pluma en vez de sobresalir a brillo pleno.

    override_contrast > 0 invierte eso, pero SOLO sobre la periferia fibrosa: el
    píxel se lleva a `humo local + override_contrast`. Sin ese piso relativo la
    mezcla por máximo lo borraría, porque el brillo sorteado (2-100) casi siempre
    queda por debajo del humo. Sobre el núcleo sigue camuflándose.
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
    píxel y devuelve True si corresponde; False si no cae ahí y hay que seguir
    con el punteado normal.

    El motivo: con punteado, el 82% de los píxeles etiquetados trayectoria sobre
    la periferia no recibía tinta — el modelo veía textura de humo fibroso
    etiquetada trayectoria, que es lo que hizo v19 (Video 3 f720 pasó de 8 a 80
    detecciones). Sobre fondo negro el punteado se mantiene: ahí cada punto salta
    con contraste ~51 contra el vacío y el trazo se lee igual.
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
    """Marca clase 2 sobre fondo. Con over_smoke también le gana al humo, pero
    solo donde ese humo es periferia fibrosa: sobre el núcleo queda oculta.

    Ahí la marca es de 1 px y no de 2, para que coincida con la tinta, que
    también va de 1 px (_SMOKE_OVERRIDE_WIDTH): con 2 px, la mitad de los píxeles
    clase 2 sobre el humo quedaban sin tinta por construcción.

    El heatmap se estampa DESPUÉS de esto, y _stamp_heatmap salta los píxeles
    clase 1, así que el gradiente no se derrama sobre el humo vecino. Importa
    porque export.py::mask_to_rgb pinta azul TODO píxel con heatmap > 0.
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


# 1 (bloque 3x3) cubre todo lo que un punto puede tocar: el ancho máximo del
# trazo, el kernel 3x3 del heatmap y los offsets de la máscara.
_PROGRESS_RADIUS = 1                         # vecindad que se fecha al anotar el progreso


def _progress_window(launch: float, duration: float) -> tuple[float, float]:
    """Ventana temporal de una trayectoria: cuándo se lanza y cuándo termina de
    recorrerse, como fracción del tramo post-ignición. Con duración 0 ocupa todo
    el tramo, que es el caso de la generación de una sola imagen.

    La duración NO depende del largo del recorrido, y eso está medido: dos arcos
    que difieren en un orden de magnitud (Video 4 f480→f660 y Video 3 f600→f780)
    tardan los mismos 3 pasos de un tramo de ~7. Tiene sentido físico — los
    fragmentos salen juntos y caen con la misma gravedad; los que llegan más
    lejos van más rápido. Atarla al largo hacía que los recorridos cortos
    aparecieran completos de golpe entre dos frames.

    El lanzamiento se adelanta si con el sorteado la trayectoria no alcanzaría a
    completarse: un pedazo de recorrido que ningún frame muestra rompería la
    equivalencia entre la unión de la secuencia y la imagen sin tiempo, que es la
    propiedad sobre la que se apoya todo sequence.py.
    """
    if duration <= 0:
        return (0.0, 1.0)
    duration = min(duration, 1.0)
    launch = min(max(launch, 0.0), 1.0 - duration)
    return (launch, launch + duration)


def _stamp_progress(progress_map: np.ndarray, py: int, px: int, value: float) -> None:
    """Registra en qué momento del recorrido se alcanza cada píxel, quedándose
    con el más temprano. Es lo que permite armar la secuencia retrocediendo (ver
    sequence.py) sin volver a dibujar la explosión.

    Que sea el momento del RECORRIDO y no el brillo es el punto: en las
    referencias una trayectoria parcial es un trazo truncado —la punta avanza y
    lo ya dibujado no cambia de brillo—, no un trazo atenuado.

    Ojo en modo ventana: guarda una sola fecha por píxel, así que donde dos
    trayectorias se cruzan la segunda pasada no se registra y el guion puede
    quedar con un hueco de pocos píxeles. No se pierde tinta (la unión de las
    ventanas cubre exactamente el acumulado, verificado).
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
    Trayectoria recta punteada desde center en la dirección angle.

    erase_prob / erase_frac_range: probabilidad de abrir un borrado y qué
    fracción del largo total borra.
    camouflage_scale: atenuación sobre los píxeles que caen en humo.
    progress_map: si se pasa, anota en qué punto del recorrido se alcanza cada
    píxel. No altera el dibujo ni consume rng.
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

    # Etiqueta: solo donde cayó tinta. Marcarla sobre el recorrido entero dejaba
    # el 85% de la clase sin nada visible, y en un frame de secuencia el 43% de
    # las colas salía 100% vacía.
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
    """Qué fracción del lazo cae dentro del lienzo, estimada muestreando la
    elipse paramétrica en vez de rasterizarla. Sirve para descartar una geometría
    antes de dibujarla."""
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
    Lazo que parte de `start` (dentro de la nube), describe una elipse completa
    —semi-eje mayor `a` = alcance, menor `b` = ancho— y cierra exactamente sobre
    su punto de partida: metralla que cae de regreso al punto de impacto.

    min_visible_fraction: si > 0, resortea la geometría hasta max_attempts veces
    hasta que esa fracción del lazo quede dentro del lienzo. Con 0.0 puede salir
    del cuadro libremente.
    max_spacing: separación máxima entre puntos, en el lanzamiento/aterrizaje.
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
    Arco abierto (media elipse) que parte de `start`, sobrevuela la nube y
    aterriza en otro punto — a diferencia de draw_returning_parabola, que cierra
    sobre el de partida. El ápice queda siempre del lado contrario al origen,
    así que nunca atraviesa la explosión.

    reach_range: semi-eje mayor `a`, en fracción de min(h, w).
    aspect_range: relación b/a, para que el arco salga achatado tipo arcoíris.
    height_factor_range: piso de `b` en múltiplos del ancho de humo, para que el
    ápice supere el borde de la pluma aunque `a` salga chico.
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


# Está bien que varios lazos se salgan del cuadro, pero no todos.
MIN_CONTAINED_FRACTION = 0.25                # fracción de parábolas forzadas a verse
MIN_VISIBLE_FRACTION = 0.5                   # cuánto de su lazo tiene que entrar en el cuadro

# Se sortea por trayectoria para tener variedad entre imágenes.
PARABOLA_MAX_SPACING_RANGE = (8, 25)         # separación máxima entre puntos del trazo, en px


def _schedule_at(progress_schedule: list[tuple[float, float]] | None,
                 i: int) -> tuple[float, float]:
    """(lanzamiento, duración) de la i-ésima trayectoria del grupo. Sin
    calendario —generación de una sola imagen— la duración es 0, o sea todo el
    tramo; irrelevante igual, porque sin progress_map no se anota nada."""
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
    """Lazos desde centros sorteados. Al menos MIN_CONTAINED_FRACTION quedan
    forzados a tener MIN_VISIBLE_FRACTION dentro del cuadro; el resto es libre."""
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

    progress_schedule trae un (lanzamiento, duración) por trayectoria en el orden
    en que se dibujan —rectas, parabólicas, sobrevuelos— y acá solo se reparte
    entre los tres grupos. Lo arma sequence.py."""
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
