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

Todas las funciones aceptan un parámetro opcional mask: si se pasa, dibuja
las trayectorias completas (todos los píxeles, sin spacing ni ráfagas) como
clase 2, solo sobre píxeles de fondo (clase 0). La prioridad de clases en la
máscara es humo > trayectoria > derrumbe > fondo.

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
_HEATMAP_KERNEL_SIZE = 3
_HEATMAP_KERNEL_SIGMA = 1.6

_MASK_OFFSETS = (-1, 0)  # trayectorias de 2 píxeles de grosor en la máscara categórica


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
_HIGHER_PRIORITY_CLASSES = (1, 3)

# Brillo del tensor de entrada por píxel de trayectoria: en vez de una curva
# determinística por distancia, cada punto dibujado sortea su intensidad de
# una gaussiana truncada al rango [_TRAJECTORY_BRIGHTNESS_RANGE], con la media
# corrida hacia el extremo blanco para que la mayoría de los puntos salgan
# claros pero con variación aleatoria punto a punto. La media se sortea una
# vez por trayectoria (uniforme en _TRAJECTORY_BRIGHTNESS_MEAN_RANGE) para que
# distintas trayectorias tengan distinto nivel de brillo entre sí.
_TRAJECTORY_BRIGHTNESS_RANGE = (50, 255)
_TRAJECTORY_BRIGHTNESS_MEAN_RANGE = (60.0, 200.0)
_TRAJECTORY_BRIGHTNESS_STD = 55.0


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
_TRAJECTORY_WIDTH_VALUES = (1, 2, 3)
_TRAJECTORY_WIDTH_PROBS = (0.85, 0.12, 0.03)
_WIDTH_OFFSETS = {
    1: (0,),
    2: (-1, 0),
    3: (-1, 0, 1),
}


def _sample_trajectory_width(rng: np.random.Generator) -> int:
    """Sortea el ancho de una trayectoria completa (categórica ponderada)."""
    return int(rng.choice(_TRAJECTORY_WIDTH_VALUES, p=_TRAJECTORY_WIDTH_PROBS))


def _paint_trajectory_pixel(
    tensor: np.ndarray,
    py: int,
    px: int,
    brightness: int,
    width: int,
    mask: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
) -> None:
    """Pinta un punto de trayectoria como un bloque de `width` x `width`
    píxeles centrado en (py, px), mezclando por máximo y clipeado al lienzo.

    Si se pasa mask, cada píxel del bloque que ya es humo (clase 1) se
    atenúa con camouflage_scale antes de mezclar, para que la trayectoria se
    camufle dentro del humo (oscurecido, ver smoke.py) en vez de sobresalir
    a brillo pleno. Fuera del humo el brillo no se toca.
    """
    h, w = tensor.shape
    for dy in _WIDTH_OFFSETS[width]:
        ny = py + dy
        if not (0 <= ny < h):
            continue
        for dx in _WIDTH_OFFSETS[width]:
            nx = px + dx
            if not (0 <= nx < w):
                continue
            pixel_brightness = brightness
            if mask is not None and mask[ny, nx] == 1:
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


def _paint_traj_mask(mask: np.ndarray, py: int, px: int, h: int, w: int) -> None:
    for dy in _MASK_OFFSETS:
        for dx in _MASK_OFFSETS:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                mask[ny, nx] = 2


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
) -> None:
    """
    Dibuja una trayectoria recta punteada desde center en la dirección angle.
    El spacing entre puntos crece cuadráticamente con la distancia al origen:
    cerca = denso, lejos = disperso. Simula desaceleración de metralla.
    erase_prob: probabilidad por oportunidad de dibujo de activar un borrado.
    erase_frac_range: (min, max) fracción del largo total a borrar por sección.
    camouflage_scale: atenuación aplicada a los píxeles que caen sobre humo
    (ver _paint_trajectory_pixel), misma escala global de smoke.py.
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

    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    pixels_drawn = 0
    max_spacing = 50
    max_dist = length if length > 0 else 1

    for py, px in points:
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
                    _paint_trajectory_pixel(tensor, py, px, brightness, width, mask, camouflage_scale)
                    pixels_drawn += 1
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
                    _paint_trajectory_pixel(tensor, py, px, brightness, width, mask, camouflage_scale)
                    pixels_drawn += 1

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

    # Mask: solo si la trayectoria tiene al menos un píxel visible en el tensor
    if mask is not None and pixels_drawn > 0:
        for py, px in points:
            if 0 <= py < h and 0 <= px < w:
                _paint_traj_mask(mask, py, px, h, w)

    # Heatmap: gradiente continuo sobre toda la línea (no solo los puntos dispersos)
    if heatmap is not None and pixels_drawn > 0:
        for py, px in points:
            if 0 <= py < h and 0 <= px < w:
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
    prev_py, prev_px = int(round(sy)), int(round(sx))
    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    pixels_drawn = 0
    all_points = []

    max_dist = 2 * a if a > 0 else 1  # distancia del punto más lejano del lazo a `start`

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
            all_points.append((spy, spx))

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
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale)
                        pixels_drawn += 1
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
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale)
                        pixels_drawn += 1

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

    # Mask: solo si la trayectoria tiene al menos un píxel visible en el tensor
    if mask is not None and pixels_drawn > 0:
        for spy, spx in all_points:
            if 0 <= spy < h and 0 <= spx < w:
                _paint_traj_mask(mask, spy, spx, h, w)

    # Heatmap: gradiente continuo sobre todo el lazo (no solo los puntos dispersos)
    if heatmap is not None and pixels_drawn > 0:
        for spy, spx in all_points:
            if 0 <= spy < h and 0 <= spx < w:
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
    prev_py, prev_px = int(round(sy)), int(round(sx))
    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    erase_remaining = 0
    grace_remaining = 0
    pixels_drawn = 0
    all_points = []

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
            all_points.append((spy, spx))

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
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale)
                        pixels_drawn += 1
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
                        _paint_trajectory_pixel(tensor, spy, spx, brightness, width, mask, camouflage_scale)
                        pixels_drawn += 1

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

    if mask is not None and pixels_drawn > 0:
        for spy, spx in all_points:
            if 0 <= spy < h and 0 <= spx < w:
                _paint_traj_mask(mask, spy, spx, h, w)

    if heatmap is not None and pixels_drawn > 0:
        for spy, spx in all_points:
            if 0 <= spy < h and 0 <= spx < w:
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
) -> None:
    """Genera múltiples trayectorias rectas desde centros aleatorios."""
    h, w = tensor.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)

    for _ in range(num_trajectories):
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
        draw_trajectory(tensor, center, angle, length, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                         camouflage_scale)


# Fracción mínima de trayectorias parabólicas que deben quedar mayormente
# dentro del lienzo por imagen (está bien que varias se salgan del cuadro,
# pero no todas) y umbral de "mayormente contenida" para cada una de ellas.
MIN_CONTAINED_FRACTION = 0.25
MIN_VISIBLE_FRACTION = 0.5

# Rango de max_spacing por trayectoria: valores bajos dan un punteado más
# cercano/denso en todo el lazo; valores altos dan el punteado más disperso
# de antes. Se sortea por trayectoria para tener variedad entre imágenes.
PARABOLA_MAX_SPACING_RANGE = (8, 25)


def draw_parabolic_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
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
        draw_returning_parabola(tensor, start, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                                 min_visible, max_spacing=max_spacing, camouflage_scale=camouflage_scale)


def draw_flyover_trajectories(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    origin: tuple[int, int],
    num_trajectories: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    camouflage_scale: float = 1.0,
) -> None:
    """Genera trayectorias de arco abierto que sobrevuelan la nube de humo."""
    for _ in range(num_trajectories):
        start = centers[rng.integers(0, len(centers))]

        if rng.random() < 0.5:
            erase_prob = rng.uniform(0.0, 0.20)
            frac_lo = rng.uniform(0.01, 0.03)
            frac_hi = rng.uniform(frac_lo, 0.05)
        else:
            erase_prob = 0.0
            frac_lo, frac_hi = 0.0, 0.0

        max_spacing = rng.uniform(*PARABOLA_MAX_SPACING_RANGE)
        draw_flyover_trajectory(tensor, start, origin, rng, mask, erase_prob, (frac_lo, frac_hi), heatmap,
                                 max_spacing=max_spacing, camouflage_scale=camouflage_scale)


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
) -> None:
    """Punto de entrada: dibuja trayectorias rectas, parabólicas y de sobrevuelo."""
    draw_straight_trajectories(tensor, centers, origin, num_straight, rng, mask, heatmap, camouflage_scale)
    draw_parabolic_trajectories(tensor, centers, origin, num_parabolic, rng, mask, heatmap, camouflage_scale)
    draw_flyover_trajectories(tensor, centers, origin, num_flyover, rng, mask, heatmap, camouflage_scale)
