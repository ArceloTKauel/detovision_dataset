"""
smoke.py - Generación de humo con textura orgánica y manchas sustractivas.

Dibuja el humo de la explosión alrededor de los centros de impacto usando
un modelo de zonas concéntricas (core, mid, outer, fringe) con distorsión
Perlin para bordes irregulares. Luego genera polígonos aleatorios que
"recortan" huecos dentro del humo, simulando manchas/huecos realistas.

Funciones:
    - draw_smoke(tensor, centers, smoke_radius, rng, mask): Dibuja el humo
      completo sobre el tensor. Usa distancia mínima a cualquier centro para
      determinar la zona, y Perlin noise para distorsionar radios y variar
      brillo. Si se pasa mask, marca los píxeles de humo como clase 1
      (y borra de la mask donde caen las manchas sustractivas).
    - draw_smoke_filaments(tensor, line, smoke_radius, rng, mask): Dibuja la
      periferia filamentosa — estrías que irradian desde la línea de tiro. Es
      lo que draw_smoke no puede dar por construcción (ver _FILAMENT_*).
    - draw_white_blobs(tensor, smoke_radius, rng, mask): Dibuja sub-nubes de
      "humo blanco" (piso de brillo 130) reutilizando draw_smoke sobre un
      centro y radio más chicos, simulando metralla/brasas incandescentes
      agrupadas. Se llama después de draw_smoke.
    - measure_smoke_width(tensor, origin, angle): Mide la distancia desde el
      origen en una dirección dada hasta encontrar un píxel negro. Usado por
      trajectories.py para que las trayectorias empiecen fuera del humo.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from canvas import sample_on_line
from perlin_noise import perlin_noise_2d

# Escala de brillo global de la explosión: sortea una vez por imagen (en
# main.py, vía sample_brightness_scale) y se aplica tanto al humo como a los
# centros camuflados debajo (ver canvas.py::draw_center), para que ninguno
# sature a blanco pleno. Acerca el rango dinámico al de las referencias sin
# binarizar (mascara_cambios_final_sinbin_*), que nunca llegan a blanco puro.
SMOKE_BRIGHTNESS_SCALE_RANGE = (0.45, 0.70)

# Rango objetivo de la clase humo (medido con pixel_inspector_gui.py): 15 a
# 255. _SMOKE_BRIGHTNESS_FLOOR evita que cualquier píxel dibujado como humo
# quede por debajo de 15 (se aplica en el clip de cada zona). El resto del
# pipeline (brightness_scale, grain, etc.) ya mantiene el grueso del humo
# bien por debajo de 255; los "flecos" definidos más abajo son la única vía
# para que un píxel puntual llegue cerca de blanco pleno, y con probabilidad
# baja a propósito (ver _HOT_FLECK_PROB).
_SMOKE_BRIGHTNESS_FLOOR = 15

# Flecos brillantes: chispas/reflejos puntuales muy poco frecuentes que
# rompen por encima del techo habitual del humo (~180) hasta casi blanco
# pleno, independientes de brightness_scale (si no, nunca podrían acercarse
# a 255 en una imagen con brightness_scale bajo). La probabilidad es baja
# para que la gran mayoría del humo se mantenga por debajo de 200.
_HOT_FLECK_PROB = 0.01
_HOT_FLECK_RANGE = (200, 255)

# Blobs de humo blanco: sub-nubes de metralla/brasas incandescentes dentro del
# humo, reutilizando draw_smoke (mismas zonas core/mid/outer/fringe + Perlin)
# sobre un centro y radio más chicos, en vez de un patrón ad hoc. El piso de
# brillo alto (_WHITE_BLOB_BRIGHTNESS_FLOOR) es lo que las distingue del humo
# gris normal: incluso la zona fringe (la de probabilidad más baja) nunca cae
# por debajo de 130, así que el borde disperso del blob queda como racimo de
# puntos brillantes en vez de humo gris tenue (ver referencia
# mascara_cambios_final_sinbin_1.png). Independiente de brightness_scale del
# humo principal, igual que los flecos: deben resaltar aunque la explosión
# haya salido "apagada".
_WHITE_BLOB_PROB = 0.7
_WHITE_BLOB_COUNT_RANGE = (1, 2)
_WHITE_BLOB_RADIUS_RATIO = (0.35, 0.6)  # fracción de smoke_radius
_WHITE_BLOB_SCALE_RANGE = (0.7, 1.0)  # análogo a brightness_scale, cercano a blanco
_WHITE_BLOB_BRIGHTNESS_FLOOR = 130


def sample_brightness_scale(rng: np.random.Generator) -> float:
    """Sortea la escala de brillo global de una explosión (una vez por imagen)."""
    return rng.uniform(*SMOKE_BRIGHTNESS_SCALE_RANGE)


def draw_smoke(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
    brightness_floor: int = _SMOKE_BRIGHTNESS_FLOOR,
    include_extras: bool = True,
) -> None:
    """
    brightness_floor: piso de brillo aplicado al clip de cada zona (ver
        _SMOKE_BRIGHTNESS_FLOOR). draw_white_blobs reusa esta función con un
        piso más alto para generar sub-nubes de "humo blanco".
    include_extras: si es False, omite flecos brillantes y manchas
        sustractivas (pensado para llamadas anidadas, como las de
        draw_white_blobs, donde esos efectos ya los aporta el humo principal).
    """
    h, w = tensor.shape

    all_cy = np.array([c[0] for c in centers], dtype=np.float64)
    all_cx = np.array([c[1] for c in centers], dtype=np.float64)

    # Región de trabajo: bounding box de todos los centros + margen del radio
    fringe_radius = smoke_radius * 1.3
    min_y = max(0, int(all_cy.min() - fringe_radius - 10))
    max_y = min(h, int(all_cy.max() + fringe_radius + 10))
    min_x = max(0, int(all_cx.min() - fringe_radius - 10))
    max_x = min(w, int(all_cx.max() + fringe_radius + 10))

    region_h = max_y - min_y
    region_w = max_x - min_x
    if region_h <= 0 or region_w <= 0:
        return

    # Grilla de coordenadas de la región
    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    # Para cada píxel, calcular la distancia mínima a cualquier centro
    min_dist = np.full(ys.shape, np.inf)
    for cy, cx in centers:
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        min_dist = np.minimum(min_dist, dist)

    region = tensor[min_y:max_y, min_x:max_x]
    mask_region = mask[min_y:max_y, min_x:max_x] if mask is not None else None

    # Perlin noise para distorsionar los bordes del humo y variar el brillo
    perlin = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.4, rng=rng, octaves=5)
    perlin_fine = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.15, rng=rng, octaves=3)

    # Grano fino: escala mucho más chica que perlin_fine, para romper la
    # superficie lisa incluso dentro del core (textura fibrosa/granulada en
    # vez de un relleno plano). Se multiplica sobre el brillo de TODAS las
    # zonas junto con brightness_scale.
    grain = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.05, rng=rng, octaves=2)
    grain_factor = 0.55 + 0.45 * grain

    # Distorsión del radio: el Perlin desplaza el borde ±30%, creando irregularidad
    radius_distortion = 1.0 + (perlin - 0.5) * 0.6
    distorted_dist = min_dist / radius_distortion

    # === ZONA CORE (0 a 25% del radio) ===
    # Zona más brillante y densa, casi sólida
    core_radius = smoke_radius * 0.25
    # Banda de transición centrada en core_radius: sin ella, el brillo (perlin_fine
    # vs perlin) y la probabilidad de dibujo (100% vs ~90%) cambian de golpe justo
    # en el borde del core, lo que se ve como una costura/corte. Dentro de la banda
    # se mezclan ambos ruidos y se mantiene dibujo 100% (sin agujeros) para que la
    # transición sea gradual.
    seam_half_width = smoke_radius * 0.06
    seam_lo = core_radius - seam_half_width
    seam_hi = core_radius + seam_half_width

    core_mask = distorted_dist < seam_lo
    core_brightness = (
        255 * (0.85 + perlin_fine[core_mask] * 0.15) * brightness_scale * grain_factor[core_mask]
    ).clip(brightness_floor, 255).astype(np.uint8)
    region[core_mask] = np.maximum(region[core_mask], core_brightness)
    if mask_region is not None:
        mask_region[core_mask] = 1

    # === ZONA DE TRANSICIÓN (seam_lo a seam_hi) ===
    seam_mask = (distorted_dist >= seam_lo) & (distorted_dist < seam_hi)
    t = (distorted_dist[seam_mask] - seam_lo) / (seam_hi - seam_lo)
    t = t * t * (3 - 2 * t)  # smoothstep: 0 en seam_lo, 1 en seam_hi
    perlin_seam = perlin_fine[seam_mask] * (1 - t) + perlin[seam_mask] * t
    seam_brightness = (
        255 * (1 - 0.3 * t) * (0.85 + perlin_seam * 0.15) * brightness_scale * grain_factor[seam_mask]
    ).clip(brightness_floor, 255).astype(np.uint8)
    region[seam_mask] = np.maximum(region[seam_mask], seam_brightness)
    if mask_region is not None:
        mask_region[seam_mask] = 1

    # === ZONA MID (seam_hi a 50% del radio) ===
    # Densidad decreciente, probabilidad de dibujar baja con la distancia
    mid_radius = smoke_radius * 0.5
    mid_mask = (distorted_dist >= seam_hi) & (distorted_dist < mid_radius)
    ratio_mid = (distorted_dist[mid_mask] - seam_hi) / (mid_radius - seam_hi)
    prob_mid = 0.9 - 0.4 * ratio_mid  # probabilidad de 90% a 50%
    brightness_mid = (
        255 * (1 - ratio_mid * 0.3) * (0.7 + perlin[mid_mask] * 0.3) * brightness_scale * grain_factor[mid_mask]
    ).clip(brightness_floor, 255).astype(np.uint8)
    mid_drawn = perlin[mid_mask] > (1 - prob_mid)
    region[mid_mask] = np.where(
        mid_drawn,
        np.maximum(region[mid_mask], brightness_mid),
        region[mid_mask],
    )
    if mask_region is not None:
        mid_indices = np.argwhere(mid_mask)
        mask_region[mid_indices[mid_drawn, 0], mid_indices[mid_drawn, 1]] = 1

    # === ZONA OUTER (50% a 100% del radio) ===
    # Humo disperso, probabilidad decae cuadráticamente
    outer_mask = (distorted_dist >= mid_radius) & (distorted_dist < smoke_radius)
    ratio_outer = (distorted_dist[outer_mask] - mid_radius) / (smoke_radius - mid_radius)
    prob_outer = 0.6 * (1 - ratio_outer) ** 2
    brightness_outer = (
        200 * (1 - ratio_outer * 0.7) * (0.6 + perlin[outer_mask] * 0.4) * brightness_scale * grain_factor[outer_mask]
    ).clip(brightness_floor, 255).astype(np.uint8)
    outer_drawn = perlin[outer_mask] > (1 - prob_outer)
    region[outer_mask] = np.where(
        outer_drawn,
        np.maximum(region[outer_mask], brightness_outer),
        region[outer_mask],
    )
    if mask_region is not None:
        outer_indices = np.argwhere(outer_mask)
        mask_region[outer_indices[outer_drawn, 0], outer_indices[outer_drawn, 1]] = 1

    # === ZONA FRINGE (100% a 130% del radio) ===
    # Borde difuso, muy baja probabilidad, partículas sueltas
    fringe_mask = (distorted_dist >= smoke_radius) & (distorted_dist < fringe_radius)
    ratio_fringe = (distorted_dist[fringe_mask] - smoke_radius) / (fringe_radius - smoke_radius)
    prob_fringe = 0.15 * (1 - ratio_fringe) ** 3
    brightness_fringe = (
        120 * (1 - ratio_fringe) * perlin[fringe_mask] * brightness_scale * grain_factor[fringe_mask]
    ).clip(brightness_floor, 255).astype(np.uint8)
    fringe_drawn = perlin[fringe_mask] > (1 - prob_fringe)
    region[fringe_mask] = np.where(
        fringe_drawn,
        np.maximum(region[fringe_mask], brightness_fringe),
        region[fringe_mask],
    )
    if mask_region is not None:
        fringe_indices = np.argwhere(fringe_mask)
        mask_region[fringe_indices[fringe_drawn, 0], fringe_indices[fringe_drawn, 1]] = 1

    if include_extras:
        # smoke_so_far / smoke_mask deben restringirse al humo real (mask_region
        # == 1), no a `region > 0`: la región de trabajo puede incluir textura de
        # terreno (draw_terrain, dibujado antes con np.maximum) que no tiene
        # nada que ver con el humo. Sin esta distinción, tanto los flecos como
        # las manchas sustractivas de más abajo terminan operando sobre terreno
        # ajeno al humo — los flecos lo salpican de chispas sin sentido y las
        # manchas lo perforan a negro puro, rompiendo la continuidad de las
        # líneas de terreno alrededor de la explosión. Si no hay mask, no hay
        # forma de distinguir humo de terreno y se cae de vuelta a `region > 0`.
        real_smoke_mask = mask_region == 1 if mask_region is not None else region > 0

        # === FLECOS BRILLANTES (chispas raras cercanas a blanco pleno) ===
        # Aplicado sobre el humo ya dibujado (todas las zonas), independiente de
        # brightness_scale a propósito: es la única vía para que un píxel de
        # humo llegue cerca de 255 aunque esta explosión haya salido "apagada".
        fleck_roll = rng.random(region.shape)
        fleck_mask = real_smoke_mask & (fleck_roll < _HOT_FLECK_PROB)
        if fleck_mask.any():
            fleck_brightness = rng.uniform(*_HOT_FLECK_RANGE, size=int(fleck_mask.sum())).astype(np.uint8)
            region[fleck_mask] = np.maximum(region[fleck_mask], fleck_brightness)

        # === MANCHAS SUSTRACTIVAS (polígonos que recortan huecos en el humo) ===
        # Se generan en zonas donde el Perlin sustractivo es bajo (< 0.45),
        # evitando el core central para no romper la forma base.
        subtractive = perlin_noise_2d(
            (region_h, region_w), scale=smoke_radius * 0.3, rng=rng, octaves=3
        )
        candidate_mask = (subtractive < 0.45) & real_smoke_mask
        candidate_mask &= distorted_dist > core_radius * 0.5  # proteger el core

        candidate_coords = np.argwhere(candidate_mask)
        if len(candidate_coords) > 0:
            num_polys = min(len(candidate_coords), rng.integers(15, 45))
            seed_indices = rng.choice(len(candidate_coords), size=num_polys, replace=False)
            seeds = candidate_coords[seed_indices]

            # Dibujar polígonos en una imagen auxiliar para luego aplicar como máscara
            poly_img = Image.new("L", (region_w, region_h), 0)
            draw = ImageDraw.Draw(poly_img)

            for seed in seeds:
                sy, sx = seed
                dist = distorted_dist[sy, sx]
                # Polígonos más pequeños cerca del core, más grandes lejos
                core_factor = float(np.clip(dist / (core_radius * 2), 0.2, 1.0))

                # Polígono irregular: vértices a ángulos ordenados con radio variable
                num_verts = rng.integers(4, 14)
                angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_verts))
                base_r = smoke_radius * rng.uniform(0.1, 0.3) * core_factor

                verts = []
                for a in angles:
                    r = base_r * rng.uniform(0.6, 1.0)  # variación del radio por vértice
                    vy = sy + r * np.sin(a)
                    vx = sx + r * np.cos(a)
                    verts.append((int(round(vx)), int(round(vy))))

                draw.polygon(verts, fill=255)

            # Aplicar máscara: donde hay polígono Y es humo real, borrar. El
            # polígono puede sobresalir un poco más allá de real_smoke_mask
            # (el radio se sortea independiente de la forma del humo); sin este
            # segundo filtro, ese sobrante perfora terreno vecino en vez de
            # limitarse al humo.
            poly_array = np.array(poly_img)
            erase_mask = (poly_array > 0) & real_smoke_mask
            region[erase_mask] = 0
            if mask_region is not None:
                mask_region[erase_mask] = 0


# ── Filamentos radiales ────────────────────────────────────────────────────
# El humo real no es un blob liso: son miles de estrías finas que irradian desde
# la línea de tiro (polvo y escombro con motion blur). draw_smoke no puede
# producirlas ni con otros parámetros — está construido como zonas de distancia,
# o sea f(min_dist), y por lo tanto no tiene noción de dirección: la isotropía
# es estructural. Estas partículas-estría aportan esa textura, y como salen de
# una línea y no de un punto, la pluma queda alargada sola.
#
# Se dibujan DESPUÉS de draw_smoke a propósito: sus manchas sustractivas operan
# sobre `mask_region == 1`, así que si los filamentos ya estuvieran marcados,
# los perforaría también.
_FILAMENT_COUNT_RANGE  = (1400, 3200)
# Arranque de la estría, en fracción de smoke_radius: NO puede ser 0. Si todas
# salen desde la línea misma, la densidad se apila ahí y el núcleo revienta a
# blanco puro, tapando toda la estructura. Repartir los arranques sobre un
# anillo distribuye el depósito y deja el núcleo para draw_smoke.
_FILAMENT_START_RATIO  = (0.10, 1.00)
_FILAMENT_LENGTH_RATIO = (0.40, 3.00)    # fracción de smoke_radius
_FILAMENT_LENGTH_SKEW  = 1.8             # >1 sesga a estrías cortas, con cola larga
_FILAMENT_BRIGHTNESS   = (60.0, 190.0)
_FILAMENT_FALLOFF_EXP  = 1.5             # cómo se apaga la estría hacia la punta
_FILAMENT_CURL         = 0.30            # deriva angular: las estrías no son rectas
# Sin esto la pluma irradia 360° parejo y sale un diente de león. Las reales son
# abanicos volcados hacia un lado (el material sale contra la cara del banco, y
# la gravedad y el viento hacen el resto). Escala el largo de la estría según su
# ángulo respecto de una dirección preferente sorteada por imagen: el largo va
# de (1-A) a (1+A) veces el nominal.
_FILAMENT_ANISOTROPY   = 0.65
# El ensanchado tiene que ser PERPENDICULAR a la estría, no isótropo. Con
# dispersión isótropa + blur (0.6/0.6) también se difumina a lo largo, la estría
# se vuelve pelusa y la coherencia de orientación cae de 0.307 a 0.234. Pero
# dejarlo en 0 da líneas duras de un píxel: eso mete demasiada energía de alta
# frecuencia y aleja la pendiente espectral de la de las referencias (-1.92
# contra -2.18). Ensanchar solo en perpendicular da ancho y suavidad sin perder
# la estructura lineal — que es lo que hace el motion blur real.
# 2.2 px medido sobre 30 semillas: es el único valor probado que mejora las DOS
# métricas a la vez respecto de no tener filamentos (coherencia 0.242 -> 0.274,
# y la pendiente espectral queda a 0.02 del objetivo en vez de a 0.04). Con
# 0.9 px la coherencia sube más (0.302) pero la pendiente se va a -1.90 contra
# -2.18 de las referencias, o sea se gana textura metiendo ruido de alta
# frecuencia que las reales no tienen.
_FILAMENT_WIDTH_PX     = 2.2
_FILAMENT_WIDTH_OFFSETS = np.array([-1.0, 0.0, 1.0])
_FILAMENT_WIDTH_WEIGHTS = np.array([0.6, 1.0, 0.6])
_FILAMENT_BLUR         = 0.0
# El acumulador se renormaliza por su propio percentil 99 a este nivel, en vez
# de depender de las constantes de arriba: así cambiar cantidad o largo altera
# la FORMA de la pluma sin volver a saturarla, que es lo que pasaba antes.
_FILAMENT_PEAK_LEVEL   = 205.0
# Umbral de DENSIDAD (no de brillo) para marcar clase humo. Se aplica sobre el
# acumulador ya renormalizado y antes de brightness_scale, así que la máscara no
# se mueve si la explosión sale más clara o más apagada.
_FILAMENT_MASK_LEVEL   = 14.0
# Fracción de estrías (no de píxeles) que se etiquetan como trayectoria (clase
# 2) en vez de humo (clase 1): las de MAYOR alcance (start + length), no un
# sorteo parejo. Motivo: en las predicciones del modelo entrenado con humo
# fibroso, algunas de las estrías largas que salen del núcleo son indistintas
# de metralla real — hoy el dataset le enseña que TODO filamento es humo, sin
# ejemplos de esa ambigüedad. Ni 0 (no la aprende) ni 1 (deja de haber
# periferia fibrosa de humo). Ver [[project_smoke_filaments]].
_FILAMENT_TRAJECTORY_FRAC = 0.12
# Piso del heatmap para los píxeles reclasificados a trayectoria. OJO: esto no
# es solo estético. detovision_segmentation/utils/dataset.py usa el canal azul
# directo como target BLANDO de CrossEntropyLoss (`trayectoria_np = B/255`), no
# como una máscara binaria — es la probabilidad que el modelo aprende a
# predecir ahí. Las trayectorias reales (trajectories.py::_HEATMAP_KERNEL)
# pican en 255 (~100% de confianza) en toda su longitud. Un piso bajo (se
# probó 60 ≈ 23%) le enseña al modelo una trayectoria "tibia" justo en los
# píxeles que se querían reforzar — señal de entrenamiento débil, no un
# problema de visualización. El piso tiene que quedar en el mismo orden que
# las trayectorias reales para que la señal sea comparable.
_FILAMENT_HEATMAP_FLOOR = 220.0


def draw_smoke_filaments(
    tensor: np.ndarray,
    line: np.ndarray,
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
    heatmap: np.ndarray | None = None,
) -> None:
    """Dibuja la periferia filamentosa del humo: estrías que irradian desde
    puntos sorteados sobre la línea de tiro `line` (ver canvas.py).

    Cada estría se rasteriza muestreando a lo largo de su recorrido. La cantidad
    de muestras se escala con el radio para que queden a menos de un píxel una
    de otra (si no, la estría sale punteada), y la amplitud por muestra se
    corrige por `largo / muestras` para que el brillo depositado por píxel no
    dependa del largo de la estría.
    """
    h, w = tensor.shape
    count = int(rng.integers(*_FILAMENT_COUNT_RANGE))

    origins = sample_on_line(line, count, rng)
    angles = rng.uniform(0, 2 * np.pi, size=count)
    base = rng.uniform(*_FILAMENT_BRIGHTNESS, size=count)
    curl = rng.normal(0, _FILAMENT_CURL, size=count)

    start = smoke_radius * rng.uniform(*_FILAMENT_START_RATIO, size=count)
    lo, hi = _FILAMENT_LENGTH_RATIO
    lengths = smoke_radius * (lo + (hi - lo) * rng.random(count) ** _FILAMENT_LENGTH_SKEW)

    # Anisotropía AXIAL alineada con la línea de tiro: cos(2*Δ) alarga las
    # estrías en ambos sentidos del eje de la línea y las acorta perpendicular,
    # con lo que la pluma se estira SOBRE la línea. Con una dirección preferente
    # sorteada al azar (cos(Δ), como estaba antes) el halo radiaba parejo y
    # redondeaba la pluma: la elongación caía de 1.88 a 1.34 contra 1.92 de las
    # referencias. Así se recupera la forma sin tener que acortar las estrías,
    # que es lo que costaba textura.
    axis_vec = line[-1] - line[0]
    axis = np.arctan2(axis_vec[0], axis_vec[1])
    lengths *= (1.0 - _FILAMENT_ANISOTROPY
                + _FILAMENT_ANISOTROPY * (1.0 + np.cos(2.0 * (angles - axis))))

    # Etiqueta por estría (no por píxel): las de mayor alcance total
    # (start + length) se marcan como candidatas a trayectoria. Se resuelve a
    # nivel de píxel más abajo (density_traj vs density) porque muchas
    # estrías se superponen en el mismo píxel y ahí sí puede haber mezcla.
    reach = start + lengths
    reach_cut = np.quantile(reach, 1.0 - _FILAMENT_TRAJECTORY_FRAC)
    is_traj = reach >= reach_cut

    samples = int(np.clip(smoke_radius * 2.5, 48, 256))
    t = np.linspace(0.0, 1.0, samples)[None, :]

    a = angles[:, None] + curl[:, None] * t
    r = start[:, None] + lengths[:, None] * t
    ys = origins[:, 0:1] + r * np.sin(a)
    xs = origins[:, 1:2] + r * np.cos(a)

    amp = base[:, None] * (1.0 - t) ** _FILAMENT_FALLOFF_EXP * (lengths[:, None] / samples)

    # Grosor: réplicas PARALELAS de la misma estría, desplazadas sobre su normal
    # (la dirección de avance es (sin a, cos a) en (y, x), así que la normal es
    # (cos a, -sin a)). El desplazamiento es constante a lo largo de cada
    # réplica: si se sortea por muestra, la estría zigzaguea y pierde la
    # orientación que es justamente lo que aporta — medido, la coherencia cae de
    # 0.306 a 0.220. Ver _FILAMENT_WIDTH_PX.
    offs = _FILAMENT_WIDTH_OFFSETS * _FILAMENT_WIDTH_PX
    ys = ys[..., None] + offs * np.cos(a)[..., None]
    xs = xs[..., None] - offs * np.sin(a)[..., None]
    amp = amp[..., None] * _FILAMENT_WIDTH_WEIGHTS

    yi = np.round(ys).astype(np.int64).ravel()
    xi = np.round(xs).astype(np.int64).ravel()
    inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
    if not inside.any():
        return

    # bincount en índices aplanados: mucho más rápido que np.add.at con este
    # volumen de muestras (cientos de miles por imagen).
    flat_idx = yi[inside] * w + xi[inside]
    density = np.bincount(flat_idx, weights=amp.ravel()[inside], minlength=h * w).reshape(h, w)

    lit = density > 0
    if not lit.any():
        return
    peak = float(np.percentile(density[lit], 99.0))
    scale = _FILAMENT_PEAK_LEVEL / max(peak, 1e-6)
    density *= scale

    brightness = np.clip(density * brightness_scale, 0, 255).astype(np.uint8)
    brightness = np.array(
        Image.fromarray(brightness).filter(ImageFilter.GaussianBlur(_FILAMENT_BLUR))
    )

    drawn = density > _FILAMENT_MASK_LEVEL
    np.maximum(tensor, np.where(drawn, brightness, 0), out=tensor)
    if mask is not None:
        # Densidad aportada solo por las estrías candidatas a trayectoria,
        # con la MISMA escala que density (misma normalización por peak), para
        # poder comparar directamente. En píxeles donde varias estrías se
        # superponen, la clase se decide por cuál aporta más de la mitad de la
        # densidad ahí — así el núcleo (muchas estrías cortas de humo
        # solapadas) se mantiene humo aunque alguna estría de trayectoria lo
        # atraviese de paso hacia su punta.
        is_traj_bc = np.broadcast_to(is_traj[:, None, None], amp.shape)
        amp_traj = np.where(is_traj_bc, amp, 0.0)
        density_traj = np.bincount(
            flat_idx, weights=amp_traj.ravel()[inside], minlength=h * w
        ).reshape(h, w)
        density_traj *= scale

        # Nunca DEGRADAR a trayectoria un píxel que draw_smoke/draw_center ya
        # marcó como humo real (prioridad humo > trayectoria, igual que en el
        # resto del pipeline): sin esto, una estría de trayectoria que de
        # paso cruza el núcleo ya sólido podía bajarlo de clase 1 a clase 2.
        already_humo = mask == 1
        traj_dominant = density_traj > (density - density_traj)
        new_traj = drawn & traj_dominant & ~already_humo
        mask[drawn & ~new_traj] = 1
        mask[new_traj] = 2

        # export.py::mask_to_rgb pinta la clase trayectoria a partir de
        # `heatmap > 0`, no de `mask == 2` directo (así funciona el gradiente
        # real de las trayectorias). Sin esto, estos píxeles quedan con
        # mask == 2 pero heatmap == 0: no matchean ninguna condición de color
        # en mask_to_rgb y salen negro puro.
        # Se usa `density` (no `brightness`) porque ya está normalizada a un
        # peak fijo (_FILAMENT_PEAK_LEVEL) independiente de brightness_scale
        # y del umbral de dibujo — brightness_scale es un concepto de tensor/
        # camuflaje que no debe apagar la confianza del heatmap, igual que en
        # trajectories.py. El piso evita que la punta de la estría (density
        # apenas sobre el umbral de "dibujado") quede casi invisible.
        if heatmap is not None:
            heatmap_value = np.clip(density, _FILAMENT_HEATMAP_FLOOR, 255.0).astype(np.uint8)
            heatmap[new_traj] = np.maximum(heatmap[new_traj], heatmap_value[new_traj])


def draw_white_blobs(
    tensor: np.ndarray,
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> None:
    """
    Dibuja una o más sub-nubes de "humo blanco" dentro del humo ya trazado,
    reutilizando draw_smoke sobre un centro y radio más chicos, con un piso
    de brillo alto (130) y sin sus propios flecos/manchas sustractivas
    (include_extras=False). Hereda la misma textura orgánica (zonas
    core/mid/outer/fringe + distorsión Perlin) que el humo normal, pero
    funciona como núcleo incandescente en vez de gris, simulando metralla o
    brasas agrupadas. No hace nada si no hay humo dibujado o si el sorteo de
    probabilidad no activa el blob en esta explosión.

    El centro se sortea entre píxeles de clase humo (mask == 1) en vez de
    `tensor > 0`: draw_terrain pinta textura de fondo con brillo bajo sobre
    buena parte del lienzo ANTES de que exista humo real, así que `tensor >
    0` también incluye terreno lejano. Sortear sobre ese conjunto podía
    plantar un blob (piso de brillo 130, marcado como clase humo) a cientos
    de píxeles de la explosión. Si no se pasa mask, no hay forma de
    distinguir humo de terreno y se cae de vuelta a `tensor > 0`.
    """
    if mask is not None:
        smoke_coords = np.argwhere(mask == 1)
    else:
        smoke_coords = np.argwhere(tensor > 0)
    if len(smoke_coords) == 0 or rng.random() > _WHITE_BLOB_PROB:
        return

    num_blobs = rng.integers(*_WHITE_BLOB_COUNT_RANGE, endpoint=True)
    for _ in range(num_blobs):
        cy, cx = smoke_coords[rng.integers(len(smoke_coords))]
        blob_radius = smoke_radius * rng.uniform(*_WHITE_BLOB_RADIUS_RATIO)
        blob_scale = rng.uniform(*_WHITE_BLOB_SCALE_RANGE)

        draw_smoke(
            tensor,
            [(int(cy), int(cx))],
            blob_radius,
            rng,
            mask=mask,
            brightness_scale=blob_scale,
            brightness_floor=_WHITE_BLOB_BRIGHTNESS_FLOOR,
            include_extras=False,
        )


def measure_smoke_width(
    tensor: np.ndarray,
    origin: tuple[int, int],
    angle: float,
) -> float:
    """
    Mide cuántos píxeles de humo hay desde el origen en la dirección dada.
    Avanza píxel a píxel hasta encontrar fondo negro o salir del lienzo.
    Usado para que las trayectorias empiecen donde termina el humo.
    """
    h, w = tensor.shape
    oy, ox = origin
    dist = 0.0
    step = 1.0

    while True:
        dist += step
        py = int(round(oy + dist * np.sin(angle)))
        px = int(round(ox + dist * np.cos(angle)))

        if py < 0 or py >= h or px < 0 or px >= w:
            break
        if tensor[py, px] == 0:
            break

    return dist
