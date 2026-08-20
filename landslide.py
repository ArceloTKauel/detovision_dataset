"""
landslide.py - Franjas de derrumbe (clase 3). DESACTIVADA: ver DRAW_LANDSLIDES.

Desprendimientos independientes de la explosión: no comparten origen ni tocan
necesariamente el humo. Las franjas de una imagen son aproximadamente paralelas
entre sí, pero cada una arranca en un punto suelto del lienzo.

Cada franja es UNA línea central —un paso del borracho con atracción leve hacia la
dirección original, ni recta ni errática— con "puntos de caída" repartidos a lo
largo. De cada uno sale una mini-recta o mini-parábola (según un alfa gaussiano)
hacia el MISMO lado de la franja, apuntando siempre al origen de la explosión, y
cada vez más corta hacia el extremo lejano.

El eje y las caídas se dibujan punteados en el tensor pero la máscara pinta el
trazo completo: misma asimetría que usaban las trayectorias antes de que la
etiqueta pasara a marcarse solo donde hay tinta (ver trajectories.py — si esta
clase se reactiva, hay que revisar esto).

La clase 3 se pinta solo sobre fondo, lo que implementa la prioridad
humo > trayectoria > derrumbe sin lógica extra mientras se llame al final. Y
ninguna franja dibuja dentro del círculo de exclusión, así que el derrumbe nunca
pasa por encima del humo ni por los huecos que dejan sus manchas sustractivas.
"""

import numpy as np

from trajectories import bresenham

_MASK_OFFSETS = (-1, 0)                      # mismo grosor que las trayectorias (2 píxeles)


def _paint_landslide_mask(mask: np.ndarray, py: int, px: int, h: int, w: int) -> None:
    for dy in _MASK_OFFSETS:
        for dx in _MASK_OFFSETS:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                mask[ny, nx] = 3


def _line_points(y0: float, x0: float, y1: float, x1: float) -> list[tuple[int, int]]:
    return bresenham(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))


def _draw_dotted(
    tensor: np.ndarray,
    mask: np.ndarray | None,
    points: list[tuple[int, int]],
    rng: np.random.Generator,
    exclude_origin: tuple[float, float] | None = None,
    exclude_radius: float = 0.0,
    max_spacing: float = 50.0,
) -> None:
    """Dibuja una polilínea ya rasterizada con el mismo patrón punteado que
    trajectories.py: spacing cuadrático más ráfagas de 1-3 píxeles. La máscara
    pinta el trazo COMPLETO, sin puntear. Nada se dibuja dentro de
    exclude_origin/exclude_radius, ni en tensor ni en mask."""
    h, w = tensor.shape
    total_len = max(len(points), 1)
    ey, ex = exclude_origin if exclude_origin is not None else (0.0, 0.0)

    def excluded(py: int, px: int) -> bool:
        """True si el punto cae dentro del radio de exclusión de la explosión."""
        return exclude_origin is not None and (py - ey) ** 2 + (px - ex) ** 2 < exclude_radius ** 2

    if mask is not None:
        for py, px in points:
            if not excluded(py, px) and 0 <= py < h and 0 <= px < w:
                _paint_landslide_mask(mask, py, px, h, w)

    pixels_since_draw = 0
    next_draw_at = 0
    burst_remaining = 0
    for i, (py, px) in enumerate(points):
        if excluded(py, px):
            continue

        if burst_remaining > 0:
            if rng.random() < 0.7 and 0 <= py < h and 0 <= px < w:
                tensor[py, px] = 255
            burst_remaining -= 1
            continue

        if pixels_since_draw >= next_draw_at:
            if 0 <= py < h and 0 <= px < w:
                tensor[py, px] = 255
            burst_remaining = rng.integers(0, 3)
            ratio = min(i / total_len, 1.0)
            spacing = ratio ** 2 * max_spacing
            next_draw_at = max(1, int(spacing + rng.uniform(-spacing * 0.3, spacing * 0.3)))
            pixels_since_draw = 0
        else:
            pixels_since_draw += 1


_WALK_TURN_STD = 0.15                        # desviación del giro aleatorio por paso (radianes)
_WALK_MEAN_REVERSION = 0.25                  # vuelta hacia "angle": 0 sin sesgo, 1 de golpe

_ALPHA_MEAN = 0.5                            # media de alfa: <0.5 recta, >=0.5 parábola
_ALPHA_STD = 0.25                            # desvío de esa gaussiana


def _parabola_points(
    y0: float, x0: float, angle: float, length: float, curvature: float, num_steps: int
) -> list[tuple[int, int]]:
    """Mini-parábola: avance lineal en angle + offset cuadrático perpendicular."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    perp_angle = angle + np.pi / 2
    cos_p, sin_p = np.cos(perp_angle), np.sin(perp_angle)

    points: list[tuple[int, int]] = []
    prev_y, prev_x = y0, x0
    for i in range(1, num_steps + 1):
        t = i / num_steps * length
        offset = curvature * t * t
        py = y0 + t * sin_a + offset * sin_p
        px = x0 + t * cos_a + offset * cos_p
        segment = _line_points(prev_y, prev_x, py, px)
        points.extend(segment[1:] if points else segment)
        prev_y, prev_x = py, px
    return points


def generate_stripe_axis(
    start_point: tuple[float, float],
    angle: float,
    length: float,
    rng: np.random.Generator,
    num_control_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Eje central de una franja: paso del borracho con sesgo — el rumbo gira al
    azar en cada paso, con atracción leve de vuelta hacia `angle`
    (Ornstein-Uhlenbeck) para que avance en esa dirección en vez de deambular.
    Retorna (points (N,2) en [y,x], t (N,) progreso normalizado)."""
    oy, ox = start_point
    n = num_control_points
    step_size = length / (n - 1)

    headings = np.empty(n)
    headings[0] = angle
    drift = 0.0
    for i in range(1, n):
        drift += -_WALK_MEAN_REVERSION * drift + rng.normal(0.0, _WALK_TURN_STD)
        headings[i] = angle + drift

    points = np.empty((n, 2))
    points[0] = [oy, ox]
    for i in range(1, n):
        py, px = points[i - 1]
        points[i, 0] = py + step_size * np.sin(headings[i - 1])
        points[i, 1] = px + step_size * np.cos(headings[i - 1])

    t = np.linspace(0.0, 1.0, n)
    return points, t


def draw_landslide_stripe(
    tensor: np.ndarray,
    start_point: tuple[float, float],
    angle: float,
    length: float,
    tooth_len_start: float,
    tooth_len_end: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    exclude_origin: tuple[float, float] | None = None,
    exclude_radius: float = 0.0,
) -> None:
    """Una franja completa: la línea central más sus puntos de caída, con largo
    máximo que va de tooth_len_start (cerca del inicio) a tooth_len_end (extremo
    lejano).

    exclude_origin cumple dos papeles: la zona sobre la que la franja nunca dibuja,
    y la referencia del "sentido" — todas las caídas apuntan hacia ese punto, esté
    la franja del lado que esté."""
    num_control_points = max(8, int(length / 60))
    axis, t = generate_stripe_axis(start_point, angle, length, rng, num_control_points)
    n = len(t)

    tooth_lens = tooth_len_start + (tooth_len_end - tooth_len_start) * t

    # Tangente local (diferencias centrales) normalizada, y su perpendicular
    tangents = np.zeros_like(axis)
    tangents[1:-1] = axis[2:] - axis[:-2]
    tangents[0] = axis[1] - axis[0]
    tangents[-1] = axis[-1] - axis[-2]
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1.0
    tangents = tangents / norms[:, None]
    perps = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # Los tramos se concatenan en una sola polilínea para que el punteado sea
    # continuo a lo largo de la franja y no se reinicie en cada punto de control.
    axis_points: list[tuple[int, int]] = []
    for i in range(n - 1):
        segment = _line_points(axis[i, 0], axis[i, 1], axis[i + 1, 0], axis[i + 1, 1])
        axis_points.extend(segment[1:] if i > 0 else segment)
    _draw_dotted(tensor, mask, axis_points, rng, exclude_origin, exclude_radius)

    # Un solo lado para TODOS los dientes, apuntando hacia la explosión. Se calcula
    # desde el punto del eje más cercano al origen, donde la perpendicular indica
    # mejor hacia dónde queda.
    if exclude_origin is not None:
        oy, ox = exclude_origin
        dists_to_origin = np.sqrt((axis[:, 0] - oy) ** 2 + (axis[:, 1] - ox) ** 2)
        closest_idx = int(np.argmin(dists_to_origin))
        to_origin = np.array([oy, ox]) - axis[closest_idx]
        stripe_side = 1.0 if np.dot(to_origin, perps[closest_idx]) > 0 else -1.0
    else:
        stripe_side = 1.0 if rng.random() < 0.5 else -1.0

    # Cada punto de caída tira su alfa para decidir si cae una mini-recta
    # (alfa < 0.5) o una mini-parábola, desde el eje hacia stripe_side.
    pos = 0.0
    while pos < 1.0:
        idx = pos * (n - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, n - 1)
        frac = idx - i0

        center = axis[i0] * (1 - frac) + axis[i1] * frac
        perp = perps[i0] * (1 - frac) + perps[i1] * frac
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = perp / perp_norm
        tooth_len_local = tooth_lens[i0] * (1 - frac) + tooth_lens[i1] * frac

        drop_len = tooth_len_local * rng.uniform(0.6, 1.0) * (1 - pos * 0.4)
        drop_angle = np.arctan2(perp[0] * stripe_side, perp[1] * stripe_side)

        alpha = np.clip(rng.normal(_ALPHA_MEAN, _ALPHA_STD), 0.0, 1.0)
        if alpha < 0.5:
            tip_point = center + perp * stripe_side * drop_len
            drop_points = _line_points(center[0], center[1], tip_point[0], tip_point[1])
        else:
            curvature = rng.uniform(0.001, 0.006) * (1 if rng.random() < 0.5 else -1)
            num_steps = max(4, int(drop_len / 8))
            drop_points = _parabola_points(center[0], center[1], drop_angle, drop_len, curvature, num_steps)

        _draw_dotted(tensor, mask, drop_points, rng, exclude_origin, exclude_radius, max_spacing=15.0)

        spacing = (3.0 + pos ** 2 * 50.0) * rng.uniform(0.7, 1.3)
        pos += spacing / length


def draw_landslides(
    tensor: np.ndarray,
    num_stripes: int,
    angle_spread: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    exclude_origin: tuple[float, float] | None = None,
    exclude_radius: float = 0.0,
) -> None:
    """num_stripes franjas independientes: comparten una dirección general
    (angle_spread es la variación máxima alrededor de un ángulo sorteado por
    imagen, para que queden aproximadamente paralelas) pero cada una arranca en un
    punto suelto del lienzo, sin relación con el origen de la explosión."""
    h, w = tensor.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)
    center_angle = rng.uniform(0, 2 * np.pi)

    for _ in range(num_stripes):
        angle = center_angle + rng.uniform(-angle_spread / 2, angle_spread / 2)
        start_point = (rng.uniform(0, h), rng.uniform(0, w))
        length = diagonal * rng.uniform(0.6, 1.3)
        tooth_len_start = rng.uniform(30.0, 90.0)
        tooth_len_end = tooth_len_start * rng.uniform(0.15, 0.35)
        draw_landslide_stripe(
            tensor, start_point, angle, length, tooth_len_start, tooth_len_end, rng, mask,
            exclude_origin, exclude_radius,
        )
