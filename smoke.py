"""
smoke.py - Dibujo de la pluma de humo, sus filamentos y sus sub-nubes.

La pluma es una unión de círculos, y son dos decisiones independientes: DÓNDE se
colocan (_SMOKE_SHAPE_MODE) y CÓMO se pintan (_SMOKE_RENDER_MODE). Encima van los
filamentos —estrías radiales desde la línea de tiro— y las sub-nubes
incandescentes.

Los modos inactivos se conservan porque v18/v20 se entrenaron con ellos; sus
constantes están marcadas como inertes y no hacen nada con la config actual.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from canvas import sample_on_line
from perlin_noise import perlin_noise_2d

# Sorteada una vez por imagen en main.py y compartida entre el humo y los centros
# camuflados debajo, para que ninguno sature: las referencias nunca llegan a 255.
SMOKE_BRIGHTNESS_SCALE_RANGE = (0.45, 0.70)  # escala de brillo global, 1 sorteo por explosión

# Rango objetivo de la clase, medido con pixel_inspector_gui.py: 15 a 255. Los
# flecos son la única vía para acercarse al techo.
_SMOKE_BRIGHTNESS_FLOOR = 15                 # valor mínimo de un píxel dibujado como humo

# ── Forma de la pluma ──────────────────────────────────────────────────────
# Colocación: "lobulos" empuja círculos desde los pozos hasta el perímetro y la
# silueta queda emergente; "casco" define primero la zona —el casco convexo de
# unos puntos sorteados alrededor de la línea— y apila círculos adentro, lo que
# separa la silueta de la textura.
#
# Render: "zonas" es un disco con zonas concéntricas, el render con el que se
# entrenó v18/v20. "aros" pone la tinta en el borde, porque un canal del dataset
# es un absdiff y en el heatmap real (ESS_F04B2072_422, f444-540) la pluma son
# arcos brillantes con el interior oscuro: el interior de un bulto no cambia entre
# frames, su borde en avance sí. El aro es la derivada de una nube que crece.
_SMOKE_SHAPE_MODE = "casco"                  # DÓNDE: "lobulos" | "casco" | "radial"
_SMOKE_RENDER_MODE = "aros"                  # CÓMO se pintan: "zonas" | "aros"

# ── Colocación por lóbulos (inerte salvo _SMOKE_SHAPE_MODE == "lobulos") ──
# El bulto tiene que ser CHICO respecto del cuerpo —si no, unos pocos se comen la
# silueta— y su centro caer SOBRE el perímetro (_SMOKE_LOBE_RING, 1.0 = el borde),
# porque con empuje desde 0 la mitad queda sepultada. El rango de radios es ancho
# a propósito: la lectura de nube la da la mezcla de tamaños. Las micro-nubes usan
# el mismo mecanismo pero adentro, así que vuelven grumoso el interior sin tocar
# la silueta.
_SMOKE_LINE_RADIUS_FRAC = 1.20               # grosor del cuerpo, en smoke_radius
_SMOKE_LOBE_COUNT = (10, 18)                 # cuántos bultos festonean el contorno
_SMOKE_LOBE_RADIUS = (0.18, 0.45)            # radio del bulto, en smoke_radius
_SMOKE_LOBE_RING = (0.75, 1.15)              # posición, en radios del cuerpo (1.0 = borde)
_SMOKE_MICRO_COUNT = (20, 45)                # micro-nubes internas: cuántas
_SMOKE_MICRO_RADIUS = (0.08, 0.22)           # radio, en smoke_radius
_SMOKE_MICRO_SPREAD = (0.0, 1.0)             # posición, en radios del cuerpo (siempre adentro)

# ── Colocación por casco (usada por "casco" y "radial") ──────────────────
# _SMOKE_HULL_OVERLAP conviene BAJO con el render de aros: es lo que apila muchos
# arcos anidados, como se ve el heatmap real (0.62 deja pocos aros gruesos y
# separados, 0.25 los anida). _SMOKE_HULL_CANDIDATES no puede quedar corto — el
# muestreo es por rechazo, y con 400 sobre un casco angosto había explosiones casi
# vacías (la semilla 77 daba 0.09% de humo).
#
# LÍMITE: el casco es convexo, así que la pluma no puede tener concavidades y la
# real sí las tiene. Si se nota, la salida no es tocar estos números sino dejar de
# usar un casco convexo.
_SMOKE_HULL_POINTS = (7, 12)                 # vértices sorteados que definen el casco
_SMOKE_HULL_ALONG = (-0.10, 1.10)            # extensión sobre la línea (0-1 = la línea)
_SMOKE_HULL_ACROSS = (0.35, 1.70)            # ancho de la pluma, en smoke_radius
_SMOKE_HULL_CANDIDATES = 2500                # candidatos antes de filtrar por "adentro"
_SMOKE_HULL_RADIUS = (0.07, 0.40)            # radio de cada círculo, en smoke_radius
_SMOKE_HULL_OVERLAP = 0.15                   # separación, en fracción de la suma de radios

# ── Cadenas dirigidas (inerte salvo _SMOKE_SHAPE_MODE == "radial") ───────
# Los círculos no van sueltos dentro del casco sino en CADENAS que salen de la
# línea de tiro hacia afuera. Da la coherencia de orientación del abanico —en las
# siete referencias la pluma es un abanico de estrías, no una nube isótropa—
# conservando el círculo como unidad de dibujo, así que sigue leyéndose como arcos
# apilados y no como pelos. El radio decrece hacia la punta: gruesa donde nace,
# fina donde llega.
_SMOKE_CHAIN_COUNT = (30, 70)                # cadenas que salen de la línea de tiro
_SMOKE_CHAIN_LENGTH = (0.5, 2.2)             # fracción de smoke_radius
_SMOKE_CHAIN_RADIUS = (0.06, 0.26)           # radio en el nacimiento
_SMOKE_CHAIN_TAPER = 0.45                    # radio en la punta, fracción del inicial
_SMOKE_CHAIN_STEP = 0.55                     # avance por círculo, en radios propios (<1 se enciman)
_SMOKE_CHAIN_CURL = 0.25                     # las cadenas no son rectas

# ── Render por aros (inerte salvo _SMOKE_RENDER_MODE == "aros") ──────────
# El interior del anillo NO va vacío (_SMOKE_RING_FILL): en el heatmap real el
# espacio entre arcos es una masa de gris texturada, no fondo negro. Su relación
# con el umbral de máscara decide si la etiqueta miente — con relleno bajo el
# interior pasaba el umbral sin iluminarse y el 45% de los píxeles clase humo
# quedaba en <= 20; con 0.55 baja a 1.4% sobre draw_smoke aislada (24% en el
# pipeline completo, la diferencia la ponen los filamentos, más tenues).
#
# Subir _SMOKE_RIM_CONTRAST sobre el render de zonas NO llega al aro: el anillo
# sale perforado por las zonas probabilísticas.
_SMOKE_RING_WIDTH = 0.14                     # sigma del aro, en radios del círculo
_SMOKE_RING_BRIGHTNESS = (90.0, 230.0)       # brillo pico del aro, sorteado por círculo
_SMOKE_RING_WOBBLE = (0.04, 0.16)            # amplitud de las 3 armónicas que ondulan el aro
_SMOKE_RING_FILL = 0.55                      # tinta adentro del aro, en fracción de su pico
_SMOKE_RING_MASK_LEVEL = 40.0                # umbral de máscara, ANTES de brightness_scale

# Cómo se cruzan dos aros. "maximo" deja una costura dura justo en el cruce, que
# es donde más aros hay; "suave" acumula por norma-p, que se comporta como el
# máximo cuando uno domina y funde cuando son parecidos. Norma-p y no suma ni
# screen: con el interior relleno, diez círculos superpuestos saturarían la pluma
# a blanco, y la norma-p está acotada por el máximo de los aportes.
_SMOKE_RING_BLEND = "suave"                  # cruce de dos aros: "maximo" | "suave" (norma-p)
_SMOKE_RING_BLEND_P = 6.0                    # exponente de la norma-p: alto tiende al máximo

# ── Sombreado interno (inerte salvo _SMOKE_RENDER_MODE == "zonas") ───────
# Medido sobre 3 semillas, brillo de los píxeles clase humo (media / desv):
#   off 83.2/43.1 · perlin 72.0/38.3 · por_fuente 89.3/48.7 · direccional 83.2/43.4
#
# "perlin" BAJA la desviación —quita variación en vez de agregarla, por eso se
# leía como veladura— y "direccional" no hace nada, por un motivo estructural y no
# de parámetros: cada bulto recibe su lado claro y su lado oscuro, así que a
# escala de pluma se promedia solo. _SMOKE_SHADE_BLUR es indistinto entre 0.0 y
# 0.12 a los tamaños actuales; queda por si los lóbulos crecen.
_SMOKE_SHADE_MODE = "por_fuente"             # "off" | "perlin" | "por_fuente" | "direccional"
_SMOKE_SHADE_SCALE = 2.5                     # tamaño de la sombra, en smoke_radius (solo "perlin")
_SMOKE_SHADE_AMPLITUDE = 0.35                # cuánto oscurece: el brillo va de (1-amp) a 1
_SMOKE_SHADE_BLUR = 0.07                     # radio que funde las fronteras entre fuentes

# ── Reborde (inerte salvo _SMOKE_RENDER_MODE == "zonas") ─────────────────
# Se calcula por lóbulo tomando el MÁXIMO, no el mínimo como la silueta, para que
# se enciendan también los bordes internos.
#
# PENDIENTE: el 0.15 se eligió contra un FRAME DE VIDEO CRUDO, donde la pluma
# tiene el interior claro. La vara que corresponde es el heatmap de diferencias,
# donde el reborde ES la señal, y contra esa vara debería ser alto y no bajo.
_SMOKE_RIM_WIDTH = 0.18                      # sigma del reborde, en distancia normalizada
_SMOKE_RIM_CONTRAST = 0.15                   # oscurece el cuerpo: 0 desactiva, 1 lo deja negro

# A ±30% (el 0.6 original) los bultos redondos se deshacen; a ±15% conservan la
# curva sin quedar de compás.
_SMOKE_EDGE_NOISE = 0.30                     # corrimiento del borde: 0.30 = ±15% del radio

# Chispas puntuales que rompen por encima del techo habitual del humo (~180).
# Independientes de brightness_scale, si no nunca se acercarían a 255 en una
# imagen apagada; la probabilidad es baja para que el grueso quede bajo 200.
_HOT_FLECK_PROB = 0.01                       # fracción de píxeles que saltan a casi blanco
_HOT_FLECK_RANGE = (200, 255)                # brillo, independiente de brightness_scale

# Manchas sustractivas: polígonos que recortan huecos dentro del humo. Con
# profundidad 0.0 el hueco se borra a negro y además pierde la etiqueta, o sea que
# mete fondo etiquetado en medio de la pluma. La palanca que decide es la
# profundidad, no la cantidad (barrido).
_SMOKE_ERASE_COUNT_RANGE = (15, 45)          # polígonos que recortan huecos, por explosión
_SMOKE_ERASE_DEPTH = 0.35                    # brillo que SOBREVIVE al recorte: 0 borra a negro

# Sub-nubes de metralla o brasas dentro del humo: draw_smoke sobre un centro y
# radio más chicos. Lo que las distingue del humo gris es el piso de brillo alto.
# Independientes de brightness_scale: resaltan aunque la explosión salga apagada.
_WHITE_BLOB_PROB = 0.7                       # probabilidad de que la explosión las lleve
_WHITE_BLOB_COUNT_RANGE = (1, 2)             # cuántas
_WHITE_BLOB_RADIUS_RATIO = (0.35, 0.6)       # fracción de smoke_radius
_WHITE_BLOB_SCALE_RANGE = (0.7, 1.0)         # su brillo, análogo a brightness_scale
_WHITE_BLOB_BRIGHTNESS_FLOOR = 130           # piso alto: es lo que las distingue del humo gris


def _casco_convexo(puntos: np.ndarray) -> np.ndarray:
    """Casco convexo por cadena monótona de Andrew; devuelve los vértices en
    orden, como (y, x). A mano y no con scipy.spatial.ConvexHull porque scipy no
    está instalado en este entorno."""
    pts = sorted({(float(p[0]), float(p[1])) for p in puntos})
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float64)

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def media_cadena(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cruz(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    inferior = media_cadena(pts)
    superior = media_cadena(reversed(pts))
    return np.array(inferior[:-1] + superior[:-1], dtype=np.float64)


def _distancia_al_casco(casco: np.ndarray, ys, xs) -> np.ndarray:
    """Distancia con signo de cada punto al borde del casco: positiva adentro.
    Para un polígono convexo alcanza con la mínima distancia a sus semiplanos, sin
    proyectar sobre cada segmento."""
    dist = np.full(np.shape(ys), np.inf, dtype=np.float64)
    n = len(casco)
    # el casco viene antihorario, así que el interior queda a la izquierda de
    # cada arista; el signo se fija con el área para no depender de eso.
    area = 0.0
    for i in range(n):
        ay, ax = casco[i]
        by, bx = casco[(i + 1) % n]
        area += ax * by - bx * ay
    orientacion = 1.0 if area > 0 else -1.0

    for i in range(n):
        ay, ax = casco[i]
        by, bx = casco[(i + 1) % n]
        ey, ex = by - ay, bx - ax
        largo = np.hypot(ey, ex)
        if largo < 1e-9:
            continue
        # distancia con signo a la recta que contiene la arista
        d = orientacion * ((ys - ay) * ex - (xs - ax) * ey) / largo
        np.minimum(dist, d, out=dist)
    return dist


def _desenfocar(campo: np.ndarray, radio: float) -> np.ndarray:
    """Desenfoque de caja sobre un campo float, por sumas acumuladas. Dos pasadas
    (kernel triangular) porque una sola deja bordes rectos visibles. No se usa el
    GaussianBlur de PIL: no acepta modo "F", y pasar por uint8 cuantizaría el campo
    justo donde importa, en variaciones de pocos por ciento."""
    r = int(max(1, round(radio / 2)))
    k = 2 * r + 1
    h, w = campo.shape
    for _ in range(2):
        p = np.pad(campo, r, mode="edge")
        c = np.pad(np.cumsum(np.cumsum(p, axis=0), axis=1), ((1, 0), (1, 0)))
        campo = (c[k:k + h, k:k + w] - c[0:h, k:k + w]
                 - c[k:k + h, 0:w] + c[0:h, 0:w]) / (k * k)
    return campo


def sample_brightness_scale(rng: np.random.Generator) -> float:
    """Sortea la escala de brillo global de una explosión (una vez por imagen)."""
    return rng.uniform(*SMOKE_BRIGHTNESS_SCALE_RANGE)


def _extras_del_humo(region, mask_region, region_h, region_w, smoke_radius,
                     rng, distorted_dist, core_radius):
    """Flecos brillantes y manchas sustractivas sobre el humo ya dibujado,
    compartidos por los dos modos de render. Los dos parámetros de distancia son
    lo único que dependía del modo: en zonas vienen del campo distorsionado, en
    aros del normalizado."""
    # Restringido al humo REAL (mask_region == 1) y no a `region > 0`: la región de
    # trabajo puede incluir textura de terreno, y sin esta distinción los flecos lo
    # salpican de chispas y las manchas lo perforan a negro, rompiendo la
    # continuidad de las líneas alrededor de la explosión. Sin mask no hay forma de
    # distinguirlos y se cae de vuelta a `region > 0`.
    real_smoke_mask = mask_region == 1 if mask_region is not None else region > 0

    # === FLECOS BRILLANTES (chispas raras cercanas a blanco pleno) ===
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
        num_polys = min(len(candidate_coords),
                        rng.integers(*_SMOKE_ERASE_COUNT_RANGE))
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

        # El segundo filtro por real_smoke_mask hace falta: el radio del polígono
        # se sortea independiente de la forma del humo, y sin esto el sobrante
        # perfora terreno vecino.
        poly_array = np.array(poly_img)
        erase_mask = (poly_array > 0) & real_smoke_mask
        # Atenúa, no borra (ver _SMOKE_ERASE_DEPTH): el hueco conserva algo de
        # tinta y sigue siendo humo, que es lo que muestra la referencia.
        region[erase_mask] = np.rint(
            region[erase_mask] * _SMOKE_ERASE_DEPTH).astype(np.uint8)
        if mask_region is not None:
            mask_region[erase_mask & (region == 0)] = 0


def draw_smoke(
    tensor: np.ndarray,
    centers: list[tuple[int, int]],
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
    brightness_floor: int = _SMOKE_BRIGHTNESS_FLOOR,
    include_extras: bool = True,
    order_map: np.ndarray | None = None,
) -> None:
    """Dibuja la pluma alrededor de los pozos y marca clase 1 en la máscara.

    order_map: gancho de la generación temporal. Se rellena con el ORDEN DE
        NACIMIENTO del círculo que posee cada píxel, normalizado a [0, 1] por su
        distancia a la fila de pozos; los no dibujados quedan en NaN. No altera la
        imagen. Ver sequence.py::_StageRecorder.finalize.
    brightness_floor: piso aplicado al clip de cada zona. draw_white_blobs reusa
        esta función con un piso más alto.
    include_extras: con False omite flecos y manchas sustractivas, para las
        llamadas anidadas donde esos efectos ya los aporta el humo principal.
    """
    h, w = tensor.shape

    all_cy = np.array([c[0] for c in centers], dtype=np.float64)
    all_cx = np.array([c[1] for c in centers], dtype=np.float64)

    # Región de trabajo: bounding box de los centros + margen. En modo casco el
    # margen tiene que cubrir lo que se aleja el casco de la línea más el radio del
    # círculo que caiga ahí; si queda corto, la pluma sale recortada por una recta.
    fringe_radius = smoke_radius * 1.3
    margen = fringe_radius
    if _SMOKE_SHAPE_MODE in ("casco", "radial"):
        margen = smoke_radius * (_SMOKE_HULL_ACROSS[1] + _SMOKE_HULL_RADIUS[1] * 1.3)
    min_y = max(0, int(all_cy.min() - margen - 10))
    max_y = min(h, int(all_cy.max() + margen + 10))
    min_x = max(0, int(all_cx.min() - margen - 10))
    max_x = min(w, int(all_cx.max() + margen + 10))

    region_h = max_y - min_y
    region_w = max_x - min_x
    if region_h <= 0 or region_w <= 0:
        return

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    # Campo NORMALIZADO: cada fuente aporta distancia/su radio y se toma el mínimo,
    # así una fuente chica da un bulto chico y la unión de todas el borde lobulado.
    def banda_de_borde(d):
        """Cuánto pesa el reborde a distancia normalizada `d` del centro de una
        fuente: máximo justo sobre su borde (d = 1) y cae a los lados."""
        return np.exp(-(((d - 1.0) / _SMOKE_RIM_WIDTH) ** 2))

    # Las tres familias de fuentes —cuerpo, lóbulos y micro-nubes— se arman en una
    # sola lista para poder anotar QUÉ fuente gana en cada píxel (`owner`), que es
    # lo que necesitan los modos de sombreado que siguen la estructura de la nube.
    line_radius = smoke_radius * _SMOKE_LINE_RADIUS_FRAC

    if _SMOKE_SHAPE_MODE in ("casco", "radial"):
        # 1. La ZONA: casco convexo de unos pocos puntos sorteados alrededor de la
        #    línea. Pocos vértices a propósito — con muchos tiende a una elipse.
        ay, ax = all_cy[0], all_cx[0]
        by, bx = all_cy[-1], all_cx[-1]
        perp = np.arctan2(by - ay, bx - ax) + np.pi / 2

        n_pts = int(rng.integers(*_SMOKE_HULL_POINTS, endpoint=True))
        t = rng.uniform(*_SMOKE_HULL_ALONG, size=n_pts)
        u = (rng.uniform(*_SMOKE_HULL_ACROSS, size=n_pts) * smoke_radius
             * rng.choice([-1.0, 1.0], size=n_pts))
        semillas = np.stack([ay + t * (by - ay) + u * np.sin(perp),
                             ax + t * (bx - ax) + u * np.cos(perp)], axis=1)
        casco = _casco_convexo(semillas)

        # 2. Los CÍRCULOS: candidatos uniformes en la caja del casco, filtrados a
        #    los que caen adentro.
        cy_lo, cy_hi = casco[:, 0].min(), casco[:, 0].max()
        cx_lo, cx_hi = casco[:, 1].min(), casco[:, 1].max()
        cand_y = rng.uniform(cy_lo, cy_hi, size=_SMOKE_HULL_CANDIDATES)
        cand_x = rng.uniform(cx_lo, cx_hi, size=_SMOKE_HULL_CANDIDATES)
        dentro = _distancia_al_casco(casco, cand_y, cand_x)
        ok = dentro > 0
        cand_y, cand_x, dentro = cand_y[ok], cand_x[ok], dentro[ok]

        r_lo, r_hi = _SMOKE_HULL_RADIUS[0] * smoke_radius, _SMOKE_HULL_RADIUS[1] * smoke_radius
        # El radio NO se ata a la distancia al borde aunque sea lo intuitivo:
        # atándolo, contra el borde se acumulan círculos diminutos que calcan la
        # recta del casco y la pluma sale con silueta de paralelogramo. Suelto, los
        # círculos de la orilla sobresalen y definen el contorno: el casco decide
        # dónde van los CENTROS, no hasta dónde llega la tinta.
        cand_r = rng.uniform(r_lo, r_hi, size=len(cand_y))

        # 3. El APILADO: de mayor a menor, aceptando un círculo solo si está
        #    bastante separado de los ya puestos. Grandes primero es lo que hace
        #    que los chicos rellenen los intersticios y se lean "apilados".
        if _SMOKE_SHAPE_MODE == "radial":
            # Cadenas desde la línea de tiro hacia afuera, recortadas al casco.
            n_cad = int(rng.integers(*_SMOKE_CHAIN_COUNT, endpoint=True))
            anc = rng.integers(0, len(centers), size=n_cad)
            angs = rng.uniform(0, 2 * np.pi, size=n_cad)
            largos = smoke_radius * rng.uniform(*_SMOKE_CHAIN_LENGTH, size=n_cad)
            r0s = smoke_radius * rng.uniform(*_SMOKE_CHAIN_RADIUS, size=n_cad)
            curls = rng.normal(0, _SMOKE_CHAIN_CURL, size=n_cad)
            cy_l, cx_l, r_l = [], [], []
            for k in range(n_cad):
                py, px = all_cy[anc[k]], all_cx[anc[k]]
                recorrido, ang = 0.0, angs[k]
                while recorrido < largos[k]:
                    t_ = recorrido / max(largos[k], 1e-6)
                    r_ = r0s[k] * (1.0 - (1.0 - _SMOKE_CHAIN_TAPER) * t_)
                    if _distancia_al_casco(casco, py, px) < -r_:
                        break
                    cy_l.append(py); cx_l.append(px); r_l.append(r_)
                    paso = max(1.0, r_ * _SMOKE_CHAIN_STEP)
                    ang += curls[k] * paso / max(smoke_radius, 1e-6)
                    py += np.sin(ang) * paso
                    px += np.cos(ang) * paso
                    recorrido += paso
            fuentes_y = np.array(cy_l); fuentes_x = np.array(cx_l); fuentes_r = np.array(r_l)
            if len(fuentes_y) == 0:
                return
        else:
            orden = np.argsort(-cand_r)
            sel_y, sel_x, sel_r = [], [], []
            for i in orden:
                py, px, pr = cand_y[i], cand_x[i], cand_r[i]
                if sel_y:
                    d = np.hypot(np.array(sel_y) - py, np.array(sel_x) - px)
                    if np.any(d < _SMOKE_HULL_OVERLAP * (np.array(sel_r) + pr)):
                        continue
                sel_y.append(py); sel_x.append(px); sel_r.append(pr)

            fuentes_y = np.array(sel_y)
            fuentes_x = np.array(sel_x)
            fuentes_r = np.array(sel_r)
            if len(fuentes_y) == 0:      # casco degenerado: no dibujar nada
                return
    else:
        # Cuerpo: la fila de pozos, todos con el mismo radio.
        fuentes_y = list(all_cy)
        fuentes_x = list(all_cx)
        fuentes_r = [line_radius] * len(centers)

        # Lóbulos del perímetro (ver _SMOKE_LOBE_RING) y micro-nubes internas
        # (ver _SMOKE_MICRO_SPREAD).
        for cuenta, radios, empujes in (
            (_SMOKE_LOBE_COUNT, _SMOKE_LOBE_RADIUS, _SMOKE_LOBE_RING),
            (_SMOKE_MICRO_COUNT, _SMOKE_MICRO_RADIUS, _SMOKE_MICRO_SPREAD),
        ):
            n = int(rng.integers(*cuenta, endpoint=True))
            anchors = rng.integers(0, len(centers), size=n)
            for anchor, frac, ang, push in zip(anchors,
                                               rng.uniform(*radios, size=n),
                                               rng.uniform(0, 2 * np.pi, size=n),
                                               rng.uniform(*empujes, size=n)):
                fuentes_y.append(all_cy[anchor] + np.sin(ang) * push * line_radius)
                fuentes_x.append(all_cx[anchor] + np.cos(ang) * push * line_radius)
                fuentes_r.append(frac * smoke_radius)

        fuentes_y = np.array(fuentes_y)
        fuentes_x = np.array(fuentes_x)
        fuentes_r = np.array(fuentes_r)

    # La silueta se acumula con MÍNIMO (la unión de las fuentes) y el reborde con
    # MÁXIMO, a propósito: con el mínimo el borde de un lóbulo tapado por otro
    # desaparece, con el máximo se enciende y dibuja las cabezas de adentro.
    normalized = np.full(ys.shape, np.inf)
    owner = np.zeros(ys.shape, dtype=np.int32)
    rim = np.zeros(ys.shape)
    for i in range(len(fuentes_y)):
        d = np.hypot(ys - fuentes_y[i], xs - fuentes_x[i]) / fuentes_r[i]
        gana = d < normalized
        normalized[gana] = d[gana]
        owner[gana] = i
        np.maximum(rim, banda_de_borde(d), out=rim)

    # Orden de nacimiento por CÍRCULO: su distancia a la fila de pozos, normalizada
    # por un percentil y no por el máximo, que lo decidiría un solo círculo lejano.
    # La unidad que nace es el círculo entero, no una franja de píxeles.
    if order_map is not None:
        d_fuente = np.full(len(fuentes_y), np.inf)
        for cy_c, cx_c in centers:
            np.minimum(d_fuente, np.hypot(fuentes_y - cy_c, fuentes_x - cx_c),
                       out=d_fuente)
        tope = max(float(np.percentile(d_fuente, 98)), 1e-6)
        orden_fuente = np.clip(d_fuente / tope, 0.0, 1.0)

    min_dist = normalized * smoke_radius
    shading = (1.0 - _SMOKE_RIM_CONTRAST) + _SMOKE_RIM_CONTRAST * rim

    region = tensor[min_y:max_y, min_x:max_x]
    mask_region = mask[min_y:max_y, min_x:max_x] if mask is not None else None

    # Perlin noise para distorsionar los bordes del humo y variar el brillo
    perlin = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.4, rng=rng, octaves=5)
    perlin_fine = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.15, rng=rng, octaves=3)

    # Grano fino: escala mucho más chica que perlin_fine, para romper la superficie
    # lisa incluso dentro del core. Se multiplica sobre TODAS las zonas junto con
    # el sombreado por reborde, así no hay que repetirlo cuatro veces.
    grain = perlin_noise_2d((region_h, region_w), scale=smoke_radius * 0.05, rng=rng, octaves=2)
    # Sombreado interno (ver _SMOKE_SHADE_MODE). Los sorteos se hacen SIEMPRE, en
    # el mismo orden y sin importar el modo, para que cambiar de modo no desplace
    # el stream del rng y las semillas sigan comparables entre modos.
    amp = _SMOKE_SHADE_AMPLITUDE
    luz = rng.uniform(0, 2 * np.pi)
    factor_cuerpo = rng.uniform(1.0 - amp, 1.0 + amp)
    factor_fuente = rng.uniform(1.0 - amp, 1.0 + amp, size=len(fuentes_y))
    # El cuerpo es UN objeto, no 40-80: sus pozos comparten factor. Si cada uno
    # llevara el suyo, la pluma quedaría con un damero a lo largo de la línea.
    factor_fuente[:len(centers)] = factor_cuerpo

    if _SMOKE_SHADE_MODE == "perlin":
        shade = perlin_noise_2d((region_h, region_w),
                                scale=smoke_radius * _SMOKE_SHADE_SCALE,
                                rng=rng, octaves=2)
        volumen = (1.0 - amp) + amp * shade
    elif _SMOKE_SHADE_MODE == "por_fuente":
        volumen = factor_fuente[owner]
    elif _SMOKE_SHADE_MODE == "direccional":
        # Cada fuente iluminada desde el mismo lado: el píxel se aclara según
        # cuánto se corre del centro de SU fuente hacia la luz. Media 1.0, así que
        # no oscurece la pluma como el perlin.
        dy = (ys - fuentes_y[owner]) / fuentes_r[owner]
        dx = (xs - fuentes_x[owner]) / fuentes_r[owner]
        proy = np.clip(dy * np.sin(luz) + dx * np.cos(luz), -1.0, 1.0)
        volumen = 1.0 + amp * proy
    else:
        volumen = np.ones(ys.shape)

    # Los modos que dependen de `owner` cambian de golpe en la frontera entre
    # fuentes vecinas, que se vería como facetas. El desenfoque las funde.
    if _SMOKE_SHADE_MODE in ("por_fuente", "direccional"):
        volumen = _desenfocar(volumen, max(1.0, smoke_radius * _SMOKE_SHADE_BLUR))

    grain_factor = (0.55 + 0.45 * grain) * shading * volumen

    if _SMOKE_RENDER_MODE == "aros":
        # Cada círculo del apilado se estampa como un ANILLO propio, sin pasar por
        # el campo de distancias: por ahí sale perforado por las zonas
        # probabilísticas de más abajo, que difuminan el borde de una nube rellena.
        tinta = np.zeros((region_h, region_w), dtype=np.float64)
        n = len(fuentes_y)
        picos = rng.uniform(*_SMOKE_RING_BRIGHTNESS, size=n)
        # Sin ondulación son circunferencias perfectas y se lee como dibujo
        # técnico; tres armónicas alcanzan para darle carácter sin dejar de cerrar.
        amp = rng.uniform(*_SMOKE_RING_WOBBLE, size=(n, 3))
        fase = rng.uniform(0, 2 * np.pi, size=(n, 3))

        for i in range(n):
            dy = ys - fuentes_y[i]
            dx = xs - fuentes_x[i]
            d = np.hypot(dy, dx)
            # solo importa la corona: fuera de ella la gaussiana ya es ~0
            cerca = d < fuentes_r[i] * 2.0
            if not cerca.any():
                continue
            ang = np.arctan2(dy, dx)
            radio = fuentes_r[i] * (1.0
                                    + amp[i, 0] * np.cos(ang + fase[i, 0])
                                    + amp[i, 1] * np.cos(2 * ang + fase[i, 1])
                                    + amp[i, 2] * np.cos(3 * ang + fase[i, 2]))
            banda = np.exp(-(((d - radio) / (_SMOKE_RING_WIDTH * fuentes_r[i])) ** 2))
            # Piso tenue adentro del aro: el heatmap real tampoco es hueco del
            # todo, tiene textura fina en el interior del bulto.
            aporte = picos[i] * np.maximum(banda, _SMOKE_RING_FILL * (d < radio))
            if _SMOKE_RING_BLEND == "suave":
                tinta += aporte ** _SMOKE_RING_BLEND_P
            else:
                np.maximum(tinta, aporte, out=tinta)

        if _SMOKE_RING_BLEND == "suave":
            tinta **= 1.0 / _SMOKE_RING_BLEND_P

        # La MÁSCARA sale de la tinta, no de la zona ocupada: con aros finos,
        # marcar el bulto entero dejaba el 63% de los píxeles clase humo
        # prácticamente en negro. El umbral se aplica ANTES de brightness_scale
        # para que la máscara no se mueva con la exposición de la explosión.
        dibujado = tinta > _SMOKE_RING_MASK_LEVEL
        brillo = np.clip(tinta * (0.55 + 0.45 * grain) * brightness_scale,
                         0, 255)
        brillo = np.where(dibujado, np.maximum(brillo, brightness_floor), 0)
        np.maximum(region, brillo.astype(np.uint8), out=region)
        if mask_region is not None:
            mask_region[dibujado] = 1
        if order_map is not None:
            order_map[min_y:max_y, min_x:max_x][dibujado] = orden_fuente[owner[dibujado]]

        if include_extras:
            # En aros no hay `core` que proteger ni campo distorsionado: se pasa el
            # normalizado, que mide lo mismo en unidades del radio de cada círculo.
            _extras_del_humo(region, mask_region, region_h, region_w,
                             smoke_radius, rng, min_dist, smoke_radius * 0.25)
        return

    # El Perlin desplaza el borde, creando irregularidad
    radius_distortion = 1.0 + (perlin - 0.5) * _SMOKE_EDGE_NOISE
    distorted_dist = min_dist / radius_distortion

    # === ZONA CORE (0 a 25% del radio) === la más brillante, casi sólida
    core_radius = smoke_radius * 0.25
    # Banda de transición centrada en core_radius: sin ella el brillo y la
    # probabilidad de dibujo cambian de golpe en el borde del core y se ve como una
    # costura. Adentro se mezclan los dos ruidos y el dibujo se mantiene al 100%.
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

    # === ZONA MID (seam_hi a 50% del radio) === la probabilidad baja con la distancia
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

    # === ZONA OUTER (50% a 100% del radio) === disperso, decae cuadráticamente
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

    # === ZONA FRINGE (100% a 130% del radio) === borde difuso, partículas sueltas
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
        _extras_del_humo(region, mask_region, region_h, region_w,
                         smoke_radius, rng, distorted_dist, core_radius)


# ── Filamentos radiales ────────────────────────────────────────────────────
# Estrías finas que irradian desde la línea de tiro: polvo y escombro con motion
# blur. draw_smoke no puede producirlas con ningún parámetro — está construido
# como zonas de distancia, f(min_dist), así que no tiene noción de dirección.
#
# Se dibujan DESPUÉS de draw_smoke: sus manchas sustractivas operan sobre la
# máscara de humo, así que si ya estuvieran marcados los perforaría.

# Bajada de 1400-3200 a un décimo: los filamentos ponían entre el 40% y el 87% de
# la clase humo (4 semillas), o sea que la pluma ERA las estrías y cualquier
# ajuste de silueta quedaba enterrado debajo. El barrido dio 100% erizo, 33%
# peluda, 10% nube con espolvoreo; se dejó 10% y no 0 para conservar algo de
# textura fibrosa, que es con lo que se entrenó v18. Solo se tocó la CANTIDAD: el
# largo, el ancho y el curl describen cómo es UNA estría y siguen midiendo bien.
#
# A vigilar: con 10x menos estrías se achica `filament_region`, y más trayectorias
# quedan ocultas en el núcleo (ver trajectories.py::_SMOKE_OVERRIDE_PROB).
_FILAMENT_COUNT_RANGE  = (140, 320)          # estrías por explosión
# NO puede arrancar en 0: si todas salen de la línea misma, la densidad se apila
# ahí y el núcleo revienta a blanco.
_FILAMENT_START_RATIO  = (0.10, 1.00)        # dónde arranca, en smoke_radius (nunca 0)
_FILAMENT_LENGTH_RATIO = (0.40, 3.00)        # fracción de smoke_radius
_FILAMENT_LENGTH_SKEW  = 1.8                 # >1 sesga a estrías cortas, con cola larga
_FILAMENT_BRIGHTNESS   = (60.0, 190.0)       # brillo en el nacimiento
_FILAMENT_FALLOFF_EXP  = 1.5                 # cómo se apaga la estría hacia la punta
_FILAMENT_CURL         = 0.30                # deriva angular: las estrías no son rectas
# Sin esto el abanico irradia 360° parejo y sale como diente de león.
_FILAMENT_ANISOTROPY   = 0.65                # vuelca el abanico: el largo va de (1-A) a (1+A)
# Medido sobre 30 semillas, 2.2 px es el único valor probado que mejora a la vez
# la coherencia de orientación (0.242 -> 0.274) y la pendiente espectral (queda a
# 0.02 del -2.18 real). Con dispersión isótropa la estría se vuelve pelusa (0.234);
# con 0.9 px la coherencia sube más pero la pendiente se va a -1.90, o sea que se
# gana textura metiendo ruido de alta frecuencia que las reales no tienen.
_FILAMENT_WIDTH_PX     = 2.2                 # ancho en px, ensanchando en perpendicular
_FILAMENT_WIDTH_OFFSETS = np.array([-1.0, 0.0, 1.0])
_FILAMENT_WIDTH_WEIGHTS = np.array([0.6, 1.0, 0.6])
_FILAMENT_BLUR         = 0.0
# El acumulador se renormaliza por su propio p99 a este nivel, así que cambiar
# cantidad o largo altera la FORMA de la pluma sin volver a saturarla.
_FILAMENT_PEAK_LEVEL   = 205.0               # nivel al que se renormaliza el p99 del acumulador
# Umbral de DENSIDAD y no de brillo, aplicado antes de brightness_scale para que
# la máscara no se mueva con la exposición.
_FILAMENT_MASK_LEVEL   = 14.0                # umbral de DENSIDAD para marcar clase humo
# Callejón sin salida (commits 1e63def/6da70d3, revertidos): reclasificar como
# trayectoria el 12% de estrías de mayor alcance. El criterio era aleatorio
# respecto de la geometría —misma textura etiquetada según un cuantil—, que es
# justo la ambigüedad que el humo fibroso vino a eliminar. Se modela con geometría
# real en trajectories.py::_SMOKE_OVERRIDE_PROB.


def draw_smoke_filaments(
    tensor: np.ndarray,
    line: np.ndarray,
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
) -> np.ndarray | None:
    """Dibuja la periferia filamentosa: estrías que irradian desde puntos
    sorteados sobre la línea de tiro.

    Cada estría se rasteriza muestreando su recorrido. Las muestras se escalan con
    el radio para que queden a menos de un píxel (si no, la estría sale punteada),
    y la amplitud se corrige por `largo / muestras` para que el brillo por píxel no
    dependa del largo.

    Retorna qué píxeles son humo SOLO por los filamentos —periferia fibrosa, no
    núcleo—, que trajectories.py usa para decidir dónde una trayectoria se ve por
    encima del humo. None si no se pasó mask, todo False si no se dibujó nada.
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

    # Anisotropía AXIAL: cos(2*Δ) alarga las estrías en ambos sentidos del eje de
    # la línea y las acorta perpendicular, así la pluma se estira SOBRE la línea.
    # Con una dirección preferente al azar (cos(Δ)) el halo radiaba parejo y la
    # elongación caía de 1.88 a 1.34, contra 1.92 de las referencias.
    axis_vec = line[-1] - line[0]
    axis = np.arctan2(axis_vec[0], axis_vec[1])
    lengths *= (1.0 - _FILAMENT_ANISOTROPY
                + _FILAMENT_ANISOTROPY * (1.0 + np.cos(2.0 * (angles - axis))))

    samples = int(np.clip(smoke_radius * 2.5, 48, 256))
    t = np.linspace(0.0, 1.0, samples)[None, :]

    a = angles[:, None] + curl[:, None] * t
    r = start[:, None] + lengths[:, None] * t
    ys = origins[:, 0:1] + r * np.sin(a)
    xs = origins[:, 1:2] + r * np.cos(a)

    amp = base[:, None] * (1.0 - t) ** _FILAMENT_FALLOFF_EXP * (lengths[:, None] / samples)

    # Grosor: réplicas PARALELAS de la misma estría, desplazadas sobre su normal.
    # El desplazamiento es constante a lo largo de cada réplica; sorteado por
    # muestra, la estría zigzaguea y pierde la orientación que es justamente lo que
    # aporta (coherencia 0.306 -> 0.220). Ver _FILAMENT_WIDTH_PX.
    offs = _FILAMENT_WIDTH_OFFSETS * _FILAMENT_WIDTH_PX
    ys = ys[..., None] + offs * np.cos(a)[..., None]
    xs = xs[..., None] - offs * np.sin(a)[..., None]
    amp = amp[..., None] * _FILAMENT_WIDTH_WEIGHTS

    yi = np.round(ys).astype(np.int64).ravel()
    xi = np.round(xs).astype(np.int64).ravel()
    inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
    if not inside.any():
        return None if mask is None else np.zeros((h, w), dtype=bool)

    # bincount sobre índices aplanados: mucho más rápido que np.add.at con este
    # volumen de muestras (cientos de miles por imagen).
    density = np.bincount(
        yi[inside] * w + xi[inside], weights=amp.ravel()[inside], minlength=h * w
    ).reshape(h, w)

    lit = density > 0
    if not lit.any():
        return None if mask is None else np.zeros((h, w), dtype=bool)
    peak = float(np.percentile(density[lit], 99.0))
    density *= _FILAMENT_PEAK_LEVEL / max(peak, 1e-6)

    brightness = np.clip(density * brightness_scale, 0, 255).astype(np.uint8)
    brightness = np.array(
        Image.fromarray(brightness).filter(ImageFilter.GaussianBlur(_FILAMENT_BLUR))
    )

    drawn = density > _FILAMENT_MASK_LEVEL
    np.maximum(tensor, np.where(drawn, brightness, 0), out=tensor)
    if mask is None:
        return None

    # Periferia fibrosa = lo que marcan los filamentos y NO estaba ya marcado como
    # humo por el núcleo. Se calcula antes de escribir la máscara, que es el único
    # momento en que los dos aportes todavía se distinguen.
    filament_only = drawn & (mask != 1)
    mask[drawn] = 1
    return filament_only


def draw_white_blobs(
    tensor: np.ndarray,
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> None:
    """Sub-nubes incandescentes dentro del humo ya trazado: draw_smoke sobre un
    centro y radio más chicos, con piso de brillo alto y sin sus propios flecos ni
    manchas. Heredan la textura del humo normal pero se leen como metralla o
    brasas agrupadas en vez de gris.

    El centro se sortea entre píxeles de clase humo (mask == 1) y no `tensor > 0`:
    draw_terrain ya pintó fondo tenue sobre buena parte del lienzo, y sortear ahí
    plantaba blobs a cientos de píxeles de la explosión. Sin mask no hay forma de
    distinguirlos y se cae de vuelta a `tensor > 0`.
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
    """Cuántos píxeles de humo hay desde el origen en una dirección: avanza hasta
    encontrar fondo negro o salir del lienzo. La usa trajectories.py para que las
    trayectorias arranquen donde termina la pluma."""
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
