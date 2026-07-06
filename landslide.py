"""
landslide.py - Generación de franjas de derrumbe (desprendimiento de tierra/rocas).

Dibuja canales independientes de la explosión (no comparten origen ni tocan
necesariamente el humo), simulando desprendimientos de tierra o rocas que
podrían haber ocurrido en cualquier parte del encuadre. Todas las franjas de
una imagen son aproximadamente paralelas entre sí (comparten una dirección
general con variación leve por franja), pero cada una arranca en un punto
aleatorio independiente del canvas. Cada franja es **una sola línea**
central (con wiggle orgánico, no un canal de 2 bordes/riel) con una textura
tipo "peine": trazos cortos perpendiculares que salen de la línea, todos
hacia el MISMO lado dentro de una franja (no alternan al azar diente por
diente), como una "T". Ese lado se elige una vez por franja, apuntando
siempre hacia el origen de la explosión (sin importar si la franja queda a
la izquierda, derecha, arriba o abajo de ella). Los dientes son más densos
y largos cerca del punto de inicio de la franja y más dispersos/cortos
hacia su extremo lejano — mismo esquema de spacing cuadrático que ya usa
trajectories.py para sus puntos, aplicado acá a densidad de dientes en vez
de puntos sobre una línea.

Todas las funciones que aceptan mask pintan clase 3 (derrumbe) solo sobre
píxeles de fondo (clase 0). Esto implementa la prioridad
humo > trayectoria > derrumbe > fondo sin lógica adicional, siempre que
draw_landslides se llame después de draw_smoke y draw_trajectories en el
pipeline de generate_explosion.

Además, ninguna franja dibuja píxeles (ni en tensor ni en mask) dentro del
círculo de exclusión de la explosión (exclude_origin/exclude_radius): el
derrumbe nunca pasa por encima del humo, incluso a través de los huecos que
dejan las manchas sustractivas del humo dentro de su propia silueta.

Funciones:
    - generate_stripe_axis(...): genera el eje central de una franja, con
      un wiggle orgánico leve (variación menor, mayormente recto).
    - draw_landslide_stripe(...): dibuja una franja completa (1 línea central
      + textura de dientes perpendiculares), con largo de diente que se
      angosta desde su punto de inicio hacia el extremo lejano.
    - draw_landslides(...): genera N franjas paralelas con puntos de inicio
      independientes, distribuidos en cualquier parte del canvas.
"""

import numpy as np

from trajectories import bresenham

_HW = 1  # half-width: mismo grosor que las trayectorias (3x3 px)


def _paint_landslide_mask(mask: np.ndarray, py: int, px: int, h: int, w: int) -> None:
    for dy in range(-_HW, _HW + 1):
        for dx in range(-_HW, _HW + 1):
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                mask[ny, nx] = 3


def _draw_line(
    tensor: np.ndarray,
    mask: np.ndarray | None,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    exclude_origin: tuple[float, float] | None = None,
    exclude_radius: float = 0.0,
) -> None:
    """
    Dibuja un segmento con Bresenham, pintando tensor y mask (clase 3).
    Si se pasa exclude_origin/exclude_radius, se saltea cualquier píxel
    dentro de ese radio (la franja nunca pasa por encima de la explosión).
    """
    h, w = tensor.shape
    points = bresenham(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    ey, ex = exclude_origin if exclude_origin is not None else (0.0, 0.0)
    for py, px in points:
        if exclude_origin is not None and (py - ey) ** 2 + (px - ex) ** 2 < exclude_radius ** 2:
            continue
        if 0 <= py < h and 0 <= px < w:
            tensor[py, px] = 255
            if mask is not None:
                _paint_landslide_mask(mask, py, px, h, w)


def generate_stripe_axis(
    start_point: tuple[float, float],
    angle: float,
    length: float,
    rng: np.random.Generator,
    num_control_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera el eje central de una franja de derrumbe como una polilínea de
    num_control_points puntos, con un wiggle orgánico leve perpendicular a
    la dirección principal (random walk suavizado con media móvil).
    Retorna (points (N,2) en [y,x], t (N,) progreso normalizado 0-1).
    """
    oy, ox = start_point
    t = np.linspace(0.0, 1.0, num_control_points)
    dist = t * length

    raw = np.cumsum(rng.uniform(-1.0, 1.0, size=num_control_points))
    kernel_size = max(3, num_control_points // 3)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(raw, kernel, mode="same")

    max_abs = np.max(np.abs(smoothed))
    if max_abs > 0:
        smoothed = smoothed / max_abs * (length * 0.03)

    perp_angle = angle + np.pi / 2
    axis_y = oy + dist * np.sin(angle) + smoothed * np.sin(perp_angle)
    axis_x = ox + dist * np.cos(angle) + smoothed * np.cos(perp_angle)

    points = np.stack([axis_y, axis_x], axis=1)
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
    """
    Dibuja una franja de derrumbe: una sola línea central (con wiggle
    orgánico) más una textura de dientes perpendiculares que salen de esa
    línea, todos hacia el MISMO lado (apuntando hacia exclude_origin, ver
    abajo), con largo máximo que va de tooth_len_start (cerca de start_point)
    a tooth_len_end (extremo lejano).
    exclude_origin/exclude_radius: zona (típicamente la explosión) sobre la
    que la franja nunca dibuja, ni en tensor ni en mask. Además, exclude_origin
    se usa como referencia para el "sentido" de la franja: todos los dientes
    apuntan hacia ese punto, sin importar de qué lado de la explosión quede
    la franja (izquierda, derecha, arriba, abajo).
    """
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

    # Eje central: una sola línea quebrada (conecta los puntos de control)
    for i in range(n - 1):
        _draw_line(tensor, mask, axis[i, 0], axis[i, 1], axis[i + 1, 0], axis[i + 1, 1], exclude_origin, exclude_radius)

    # Sentido de la franja: un solo lado para TODOS los dientes, apuntando
    # hacia exclude_origin (la explosión). Se calcula desde el punto del eje
    # más cercano al origen (ahí la perpendicular indica mejor "hacia dónde"
    # queda la explosión respecto a la franja).
    if exclude_origin is not None:
        oy, ox = exclude_origin
        dists_to_origin = np.sqrt((axis[:, 0] - oy) ** 2 + (axis[:, 1] - ox) ** 2)
        closest_idx = int(np.argmin(dists_to_origin))
        to_origin = np.array([oy, ox]) - axis[closest_idx]
        stripe_side = 1.0 if np.dot(to_origin, perps[closest_idx]) > 0 else -1.0
    else:
        stripe_side = 1.0 if rng.random() < 0.5 else -1.0

    # Textura "peine": todos los dientes salen del eje hacia stripe_side (no
    # ambos lados), como una "T" — el eje queda como una única línea, sin riel.
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

        tooth_len = tooth_len_local * rng.uniform(0.6, 1.0) * (1 - pos * 0.4)
        tip_point = center + perp * stripe_side * tooth_len
        _draw_line(tensor, mask, center[0], center[1], tip_point[0], tip_point[1], exclude_origin, exclude_radius)

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
    """
    Genera num_stripes franjas de derrumbe independientes entre sí: comparten
    una dirección general (angle_spread es la variación máxima entre franjas
    alrededor de un ángulo central aleatorio por imagen, para que queden
    aproximadamente paralelas), pero cada una arranca en un punto aleatorio
    distinto, distribuido en cualquier parte del canvas sin relación con el
    origen de la explosión. exclude_origin/exclude_radius acotan la zona de
    la explosión sobre la que ninguna franja puede dibujar (ver docstring de
    módulo).
    """
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
