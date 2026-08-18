"""
smoke.py - Dibujo de la pluma de humo, sus filamentos y sus sub-nubes.

La pluma es una unión de círculos. Dónde se colocan lo decide
_SMOKE_SHAPE_MODE y cómo se pintan _SMOKE_RENDER_MODE; ver la sección de
constantes.

Funciones:
    - draw_smoke(tensor, centers, smoke_radius, rng, mask): dibuja la pluma
      alrededor de los pozos de la línea de tiro y marca clase 1 en la máscara.
    - draw_smoke_filaments(tensor, line, smoke_radius, rng, mask): estrías
      radiales desde la línea de tiro. Devuelve qué píxeles son humo solo por
      los filamentos, que trajectories.py usa para decidir qué trayectoria se ve
      sobre la pluma.
    - draw_white_blobs(tensor, smoke_radius, rng, mask): sub-nubes
      incandescentes dentro del humo, reutilizando draw_smoke con un piso de
      brillo alto.
    - measure_smoke_width(tensor, origin, angle): hasta dónde llega el humo en
      una dirección. La usa trajectories.py para arrancar fuera de la pluma.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from canvas import sample_on_line
from perlin_noise import perlin_noise_2d

# Escala de brillo global de la explosión, sorteada una vez por imagen en
# main.py. La comparten el humo y los centros camuflados debajo, para que
# ninguno sature a blanco pleno: las referencias sin binarizar nunca llegan a 255.
SMOKE_BRIGHTNESS_SCALE_RANGE = (0.45, 0.70)

# Piso de brillo de un píxel de humo. El rango objetivo de la clase, medido con
# pixel_inspector_gui.py, es 15 a 255; los flecos son la única vía para acercarse
# al techo.
_SMOKE_BRIGHTNESS_FLOOR = 15

# ── Forma de la pluma ──────────────────────────────────────────────────────
# La pluma es una unión de círculos. Son dos decisiones independientes: DÓNDE se
# colocan y CÓMO se pintan.
#
# Colocación (_SMOKE_SHAPE_MODE):
#   "lobulos"  círculos empujados desde los pozos de la línea de tiro hasta el
#              perímetro del cuerpo, más micro-nubes adentro. La silueta es
#              emergente, no está definida en ningún lado.
#   "casco"    se define primero la zona —el casco convexo de unos puntos
#              sorteados alrededor de la línea— y se apilan círculos adentro,
#              grandes primero. Separa la silueta de la textura.
#
# Render (_SMOKE_RENDER_MODE):
#   "zonas"    disco con zonas concéntricas (core/mid/outer/fringe). Es el
#              render con el que se entrenó v18/v20.
#   "aros"     anillo: la tinta vive en el borde. Un canal del dataset es un
#              absdiff, y en el heatmap real
#              (video_diff_heatmap_blocks_outputs/ESS_F04B2072_422, f444-540) la
#              pluma son arcos brillantes con el interior oscuro — el interior de
#              un bulto no cambia entre frames consecutivos, su borde en avance
#              sí. El aro no es un efecto estético: es lo que produce la derivada
#              de una nube que crece.
_SMOKE_SHAPE_MODE = "casco"
_SMOKE_RENDER_MODE = "aros"

# Colocación por lóbulos. Dos condiciones que la primera versión no cumplía: el
# bulto tiene que ser CHICO respecto del cuerpo —si no, unos pocos se comen la
# silueta— y su centro caer SOBRE el perímetro (_SMOKE_LOBE_RING, 1.0 = justo en
# el borde), porque con empuje desde 0 la mitad queda sepultada sin aportar al
# contorno. El rango de radios es ancho a propósito: la lectura de nube la da la
# mezcla de tamaños.
#
# Las micro-nubes usan el mismo mecanismo pero se plantan ADENTRO, así que no
# tocan la silueta: vuelven grumoso un interior que si no es un relleno parejo.
_SMOKE_LINE_RADIUS_FRAC = 1.20
_SMOKE_LOBE_COUNT = (10, 18)
_SMOKE_LOBE_RADIUS = (0.18, 0.45)
_SMOKE_LOBE_RING = (0.75, 1.15)
_SMOKE_MICRO_COUNT = (20, 45)
_SMOKE_MICRO_RADIUS = (0.08, 0.22)
_SMOKE_MICRO_SPREAD = (0.0, 1.0)

# Colocación por casco. _SMOKE_HULL_OVERLAP es cuánto tienen que separarse dos
# círculos, en fracción de la suma de radios: 0.72 se desarma en burbujas, 0.45
# masa coherente, 0.32 coherente y festoneada; por debajo vuelve a ser mancha.
#
# El radio NO se ata a la distancia al borde aunque sea lo intuitivo: atándolo,
# contra el borde se acumulan círculos diminutos que calcan la recta del casco y
# la pluma sale con silueta de polígono.
#
# LÍMITE: el casco es convexo, así que la pluma no puede tener concavidades y la
# real sí las tiene. Si se nota, la salida no es tocar estos números sino dejar
# de usar un casco convexo.
_SMOKE_HULL_POINTS = (7, 12)
_SMOKE_HULL_ALONG = (-0.10, 1.10)
_SMOKE_HULL_ACROSS = (0.35, 1.70)
_SMOKE_HULL_CANDIDATES = 400
_SMOKE_HULL_RADIUS = (0.07, 0.40)
_SMOKE_HULL_OVERLAP = 0.62

# Render por aros. El umbral de máscara va sobre la tinta ANTES de
# brightness_scale (mismo criterio que _FILAMENT_MASK_LEVEL) y tiene que quedar
# por encima de lo que aporta el relleno, o el relleno etiqueta el interior sin
# iluminarlo: con umbral 18 el 45% de los píxeles de clase humo quedaba en valor
# <= 20, con 70 queda el 15%. Ese 15% es el precio del aro —un anillo tiene mucho
# más borde que un disco— contra 1.4% del render de zonas.
#
# Subir _SMOKE_RIM_CONTRAST sobre el render de zonas NO es una alternativa para
# llegar al aro: el anillo sale perforado por las zonas probabilísticas.
_SMOKE_RING_WIDTH = 0.14
_SMOKE_RING_BRIGHTNESS = (90.0, 230.0)
_SMOKE_RING_WOBBLE = (0.04, 0.16)
_SMOKE_RING_FILL = 0.12
_SMOKE_RING_MASK_LEVEL = 70.0

# Sombreado interno del render de zonas. Medido sobre 3 semillas, brillo de los
# píxeles de clase humo:
#
#   off                 media 83.2   desv 43.1
#   perlin (0.35)             72.0        38.3   <- oscurece Y aplana
#   por_fuente (0.35)         89.3        48.7
#   direccional (0.35)        83.2        43.4   <- no hace nada
#
# "perlin" BAJABA la desviación: en vez de agregar variación la quitaba, por eso
# se leía como veladura. "direccional" es un callejón sin salida estructural y no
# de parámetros — cada bulto recibe su lado claro y su lado oscuro, así que a
# escala de pluma se promedia solo, y los 40-80 pozos del cuerpo dejan a cada
# píxel cerca del centro de SU pozo, donde la proyección vale ~0.
#
# _SMOKE_SHADE_BLUR funde las fronteras entre fuentes vecinas; medido es
# indistinto entre 0.0 y 0.12 a los tamaños actuales, queda como seguro por si
# los lóbulos crecen.
_SMOKE_SHADE_MODE = "por_fuente"
_SMOKE_SHADE_SCALE = 2.5
_SMOKE_SHADE_AMPLITUDE = 0.35
_SMOKE_SHADE_BLUR = 0.07

# Reborde de cada lóbulo en el render de zonas. Se calcula por lóbulo y se toma
# el MÁXIMO, no el mínimo como la silueta, para que se enciendan también los
# bordes internos; con el mínimo solo se vería el contorno exterior de la unión.
# _SMOKE_RIM_CONTRAST es cuánto se oscurece el cuerpo: 0 lo desactiva, 1 lo deja
# en negro.
#
# OJO, PENDIENTE: el 0.15 se eligió contra un FRAME DE VIDEO CRUDO, donde la
# pluma tiene el interior claro. La vara que corresponde es el heatmap de
# diferencias, donde el reborde ES la señal, y contra esa vara el valor debería
# ser alto y no bajo.
_SMOKE_RIM_WIDTH = 0.18
_SMOKE_RIM_CONTRAST = 0.15

# Cuánto corre el Perlin el borde de la pluma. A ±30% (el 0.6 original) los
# bultos redondos se deshacen; a ±15% conservan la curva sin quedar de compás.
_SMOKE_EDGE_NOISE = 0.30

# Flecos brillantes: chispas/reflejos puntuales muy poco frecuentes que
# rompen por encima del techo habitual del humo (~180) hasta casi blanco
# pleno, independientes de brightness_scale (si no, nunca podrían acercarse
# a 255 en una imagen con brightness_scale bajo). La probabilidad es baja
# para que la gran mayoría del humo se mantenga por debajo de 200.
_HOT_FLECK_PROB = 0.01
_HOT_FLECK_RANGE = (200, 255)

# Manchas sustractivas: polígonos que recortan huecos dentro del humo.
# _SMOKE_ERASE_DEPTH es qué fracción del brillo SOBREVIVE al recorte; 0.0 borra a
# negro y además limpia la etiqueta, o sea que mete fondo etiquetado en medio de
# la pluma. Barrido: la palanca que decide es la profundidad, no la cantidad.
_SMOKE_ERASE_COUNT_RANGE = (15, 45)
_SMOKE_ERASE_DEPTH = 0.35

# Sub-nubes de metralla o brasas incandescentes dentro del humo. Reutilizan
# draw_smoke sobre un centro y radio más chicos; lo que las distingue del humo
# gris es el piso de brillo alto, que deja hasta su borde disperso como racimo de
# puntos brillantes. Independientes de brightness_scale: deben resaltar aunque la
# explosión salga apagada.
_WHITE_BLOB_PROB = 0.7
_WHITE_BLOB_COUNT_RANGE = (1, 2)
_WHITE_BLOB_RADIUS_RATIO = (0.35, 0.6)  # fracción de smoke_radius
_WHITE_BLOB_SCALE_RANGE = (0.7, 1.0)
_WHITE_BLOB_BRIGHTNESS_FLOOR = 130


def _casco_convexo(puntos: np.ndarray) -> np.ndarray:
    """Casco convexo por cadena monótona de Andrew. O(n log n), sin dependencias.

    No se usa scipy.spatial.ConvexHull porque scipy no está instalado en este
    entorno (verificado). Devuelve los vértices en orden, como (y, x).
    """
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

    Para un polígono convexo alcanza con la mínima distancia con signo a sus
    semiplanos, sin necesidad de proyectar sobre cada segmento.
    """
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
    """Desenfoque de caja sobre un campo float, por sumas acumuladas.

    Se aplica dos veces (un kernel triangular) porque una sola pasada deja bordes
    rectos visibles. No se usa ImageFilter.GaussianBlur de PIL porque no acepta
    modo "F" y convertir a uint8 cuantizaría el campo justo donde importa, que es
    en variaciones suaves de pocos por ciento.
    """
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
    """Flecos brillantes y manchas sustractivas sobre el humo ya dibujado.

    Estaba en línea dentro de draw_smoke; se sacó acá para que los dos modos de
    render —zonas y aros— compartan el mismo código en vez de duplicarlo. Los dos
    parámetros de distancia son lo único que dependía del modo: en zonas vienen
    del campo distorsionado, en aros del campo normalizado sin distorsionar.
    """
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

        # Aplicar máscara: donde hay polígono Y es humo real, borrar. El
        # polígono puede sobresalir un poco más allá de real_smoke_mask
        # (el radio se sortea independiente de la forma del humo); sin este
        # segundo filtro, ese sobrante perfora terreno vecino en vez de
        # limitarse al humo.
        poly_array = np.array(poly_img)
        erase_mask = (poly_array > 0) & real_smoke_mask
        # Atenuar, no necesariamente borrar: ver _SMOKE_ERASE_DEPTH. Con 0.0
        # esto es exactamente lo de antes (el píxel queda en 0 y por lo tanto
        # se le limpia la etiqueta); con un piso mayor el hueco conserva algo
        # de tinta y sigue siendo humo, que es lo que muestra la referencia.
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

    # Región de trabajo: bounding box de todos los centros + margen del radio.
    # En modo casco el margen tiene que cubrir lo que se aleja el casco de la
    # línea (_SMOKE_HULL_ACROSS) más el radio del círculo que se plante ahí y su
    # propia zona fringe; si queda corto, la pluma sale recortada por una recta.
    fringe_radius = smoke_radius * 1.3
    margen = fringe_radius
    if _SMOKE_SHAPE_MODE == "casco":
        margen = smoke_radius * (_SMOKE_HULL_ACROSS[1] + _SMOKE_HULL_RADIUS[1] * 1.3)
    min_y = max(0, int(all_cy.min() - margen - 10))
    max_y = min(h, int(all_cy.max() + margen + 10))
    min_x = max(0, int(all_cx.min() - margen - 10))
    max_x = min(w, int(all_cx.max() + margen + 10))

    region_h = max_y - min_y
    region_w = max_x - min_x
    if region_h <= 0 or region_w <= 0:
        return

    # Grilla de coordenadas de la región
    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]

    # Campo NORMALIZADO: cada fuente aporta distancia/su radio y se toma el
    # mínimo, así una fuente chica genera un bulto chico y la unión de todas da el
    # borde lobulado. Se escala a smoke_radius para que las zonas de más abajo
    # sigan expresadas en fracciones del radio.
    def banda_de_borde(d):
        """Cuánto pesa el reborde a distancia normalizada `d` del centro de una
        fuente: máximo justo sobre su borde (d = 1) y cae a los lados."""
        return np.exp(-(((d - 1.0) / _SMOKE_RIM_WIDTH) ** 2))

    # Las tres familias de fuentes —cuerpo, lóbulos del perímetro y micro-nubes
    # internas— se arman primero en una sola lista y después se recorren juntas.
    # Antes eran tres bucles separados; se unificaron para poder anotar QUÉ fuente
    # gana en cada píxel (`owner`), que es lo que necesitan los modos de sombreado
    # que siguen a la estructura de la nube en vez de pintarle ruido encima.
    line_radius = smoke_radius * _SMOKE_LINE_RADIUS_FRAC

    if _SMOKE_SHAPE_MODE == "casco":
        # 1. La ZONA: casco convexo de unos pocos puntos sorteados alrededor de
        #    la línea de tiro. Pocos vértices a propósito — con muchos el casco
        #    tiende a una elipse y se pierde el carácter.
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
        #    los que caen adentro. El radio sigue a la distancia al borde, así
        #    que quedan grandes en el medio y chicos contra el contorno.
        cy_lo, cy_hi = casco[:, 0].min(), casco[:, 0].max()
        cx_lo, cx_hi = casco[:, 1].min(), casco[:, 1].max()
        cand_y = rng.uniform(cy_lo, cy_hi, size=_SMOKE_HULL_CANDIDATES)
        cand_x = rng.uniform(cx_lo, cx_hi, size=_SMOKE_HULL_CANDIDATES)
        dentro = _distancia_al_casco(casco, cand_y, cand_x)
        ok = dentro > 0
        cand_y, cand_x, dentro = cand_y[ok], cand_x[ok], dentro[ok]

        r_lo, r_hi = _SMOKE_HULL_RADIUS[0] * smoke_radius, _SMOKE_HULL_RADIUS[1] * smoke_radius
        cand_r = np.clip(dentro, r_lo, r_hi)

        # 3. El APILADO: de mayor a menor, aceptando un círculo solo si está lo
        #    bastante separado de los ya puestos. Grandes primero es lo que hace
        #    que los chicos terminen rellenando los intersticios, que es la
        #    lectura de "apilados" y no de "sorteados".
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
    # MÁXIMO: son dos operaciones distintas a propósito. Con el mínimo, el borde
    # de un lóbulo tapado por otro desaparece; con el máximo se enciende igual, y
    # eso es lo que dibuja las cabezas de adentro de la nube.
    normalized = np.full(ys.shape, np.inf)
    owner = np.zeros(ys.shape, dtype=np.int32)
    rim = np.zeros(ys.shape)
    for i in range(len(fuentes_y)):
        d = np.hypot(ys - fuentes_y[i], xs - fuentes_x[i]) / fuentes_r[i]
        gana = d < normalized
        normalized[gana] = d[gana]
        owner[gana] = i
        np.maximum(rim, banda_de_borde(d), out=rim)

    min_dist = normalized * smoke_radius
    shading = (1.0 - _SMOKE_RIM_CONTRAST) + _SMOKE_RIM_CONTRAST * rim

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
    # El sombreado por reborde entra acá, multiplicado junto con el grano, para
    # que llegue a TODAS las zonas (core, seam, mid, outer) sin repetirlo cuatro
    # veces. Ver _SMOKE_RIM_CONTRAST.
    # Sombreado interno. Ver _SMOKE_SHADE_MODE para qué hace cada modo y por qué.
    # Los sorteos se hacen SIEMPRE, en el mismo orden y sin importar el modo, para
    # que cambiar de modo no desplace el stream del rng y las semillas sigan
    # siendo comparables entre modos.
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
        # Cada fuente iluminada desde un mismo lado: el píxel se aclara según
        # cuánto se corre del centro de SU fuente hacia la luz. La media queda en
        # 1.0, así que no oscurece la pluma como el perlin.
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
        # Cada círculo del apilado se estampa como un ANILLO. No se deriva del
        # campo de distancias con _SMOKE_RIM_CONTRAST: por ahí el anillo sale
        # perforado, porque después pasa por las zonas probabilísticas de más
        # abajo, que están para difuminar el borde de una nube rellena.
        tinta = np.zeros((region_h, region_w), dtype=np.float64)
        n = len(fuentes_y)
        picos = rng.uniform(*_SMOKE_RING_BRIGHTNESS, size=n)
        # Ondulación por armónicas: sin esto son circunferencias perfectas y se
        # lee como un dibujo técnico. Tres armónicas alcanzan para que el arco
        # tenga carácter sin dejar de cerrar.
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
            # Un piso tenue adentro del aro: el heatmap real tampoco es hueco del
            # todo, tiene textura fina en el interior del bulto.
            aporte = picos[i] * np.maximum(banda, _SMOKE_RING_FILL * (d < radio))
            np.maximum(tinta, aporte, out=tinta)

        # La MÁSCARA sale de la tinta, no de la zona ocupada. Es la diferencia
        # que hace que este modo no repita el problema de etiqueta sin evidencia:
        # con aros finos, marcar el bulto entero dejaba el 63% de los píxeles de
        # clase humo prácticamente en negro. El umbral se aplica ANTES de
        # brightness_scale, igual que _FILAMENT_MASK_LEVEL, para que la máscara no
        # se mueva si la explosión sale más clara o más apagada.
        dibujado = tinta > _SMOKE_RING_MASK_LEVEL
        brillo = np.clip(tinta * (0.55 + 0.45 * grain) * brightness_scale,
                         0, 255)
        brillo = np.where(dibujado, np.maximum(brillo, brightness_floor), 0)
        np.maximum(region, brillo.astype(np.uint8), out=region)
        if mask_region is not None:
            mask_region[dibujado] = 1

        if include_extras:
            # En aros no hay `core` que proteger ni campo distorsionado: se le
            # pasa el campo normalizado, que mide lo mismo en unidades del
            # radio de cada círculo.
            _extras_del_humo(region, mask_region, region_h, region_w,
                             smoke_radius, rng, min_dist, smoke_radius * 0.25)
        return

    # Distorsión del radio: el Perlin desplaza el borde, creando irregularidad
    radius_distortion = 1.0 + (perlin - 0.5) * _SMOKE_EDGE_NOISE
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
        _extras_del_humo(region, mask_region, region_h, region_w,
                         smoke_radius, rng, distorted_dist, core_radius)


# ── Filamentos radiales ────────────────────────────────────────────────────
# Estrías finas que irradian desde la línea de tiro: polvo y escombro con motion
# blur. draw_smoke no puede producirlas ni con otros parámetros — está construido
# como zonas de distancia, o sea f(min_dist), y por lo tanto no tiene noción de
# dirección. Como salen de una línea y no de un punto, la pluma queda alargada.
#
# Se dibujan DESPUÉS de draw_smoke: sus manchas sustractivas operan sobre la
# máscara de humo, así que si los filamentos ya estuvieran marcados los perforaría.

# Cantidad de estrías por explosión. Bajada de 1400-3200 a un décimo: medido
# sobre 4 semillas, los filamentos ponían entre el 40% y el 87% de la clase humo,
# o sea que la pluma ERA las estrías y cualquier ajuste de la silueta quedaba
# enterrado debajo. Con el cuerpo en 1.20, el barrido dio 100% erizo, 33% peluda,
# 10% nube con espolvoreo, 0% nube limpia; se dejó 10% y no 0 para conservar algo
# de textura fibrosa, que es con lo que se entrenó v18.
#
# Se toca solo la CANTIDAD: el largo, el ancho y el curl describen cómo es UNA
# estría y siguen midiendo bien.
#
# A vigilar: draw_smoke_filaments devuelve `filament_region`, que trajectories.py
# usa para decidir qué trayectoria se ve sobre la pluma. Con 10x menos estrías esa
# periferia se achica y más trayectorias quedan ocultas en el núcleo.
_FILAMENT_COUNT_RANGE  = (140, 320)
# Arranque de la estría en fracción de smoke_radius. NO puede ser 0: si todas
# salen de la línea misma, la densidad se apila ahí y el núcleo revienta a blanco.
_FILAMENT_START_RATIO  = (0.10, 1.00)
_FILAMENT_LENGTH_RATIO = (0.40, 3.00)    # fracción de smoke_radius
_FILAMENT_LENGTH_SKEW  = 1.8             # >1 sesga a estrías cortas, con cola larga
_FILAMENT_BRIGHTNESS   = (60.0, 190.0)
_FILAMENT_FALLOFF_EXP  = 1.5             # cómo se apaga la estría hacia la punta
_FILAMENT_CURL         = 0.30            # deriva angular: las estrías no son rectas
# Vuelca el abanico hacia un lado en vez de irradiar 360° parejo, que salía como
# diente de león. Escala el largo según el ángulo: de (1-A) a (1+A) del nominal.
_FILAMENT_ANISOTROPY   = 0.65
# Ancho de la estría, en píxeles, ensanchando PERPENDICULAR y no isótropo. Medido
# sobre 30 semillas, 2.2 px es el único valor probado que mejora a la vez la
# coherencia de orientación (0.242 -> 0.274) y la pendiente espectral (queda a
# 0.02 del -2.18 de las referencias). Con dispersión isótropa la estría se vuelve
# pelusa y la coherencia cae a 0.234; con 0.9 px la coherencia sube más pero la
# pendiente se va a -1.90, o sea que se gana textura metiendo ruido de alta
# frecuencia que las reales no tienen.
_FILAMENT_WIDTH_PX     = 2.2
_FILAMENT_WIDTH_OFFSETS = np.array([-1.0, 0.0, 1.0])
_FILAMENT_WIDTH_WEIGHTS = np.array([0.6, 1.0, 0.6])
_FILAMENT_BLUR         = 0.0
# El acumulador se renormaliza por su propio percentil 99 a este nivel, así que
# cambiar cantidad o largo altera la FORMA de la pluma sin volver a saturarla.
_FILAMENT_PEAK_LEVEL   = 205.0
# Umbral de DENSIDAD (no de brillo) para marcar clase humo, aplicado antes de
# brightness_scale para que la máscara no se mueva con la exposición.
_FILAMENT_MASK_LEVEL   = 14.0
# Callejón sin salida ya probado (commits 1e63def/6da70d3, revertidos):
# reclasificar como trayectoria el 12% de estrías de mayor alcance. El criterio
# era aleatorio respecto de la geometría —una estría de humo y una de trayectoria
# son la misma textura, etiquetadas según un cuantil— que es justo la ambigüedad
# que el humo fibroso vino a eliminar. La ambigüedad se modela con geometría real
# en trajectories.py::_SMOKE_OVERRIDE_PROB.


def draw_smoke_filaments(
    tensor: np.ndarray,
    line: np.ndarray,
    smoke_radius: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    brightness_scale: float = 1.0,
) -> np.ndarray | None:
    """Dibuja la periferia filamentosa del humo: estrías que irradian desde
    puntos sorteados sobre la línea de tiro `line` (ver canvas.py).

    Cada estría se rasteriza muestreando a lo largo de su recorrido. La cantidad
    de muestras se escala con el radio para que queden a menos de un píxel una
    de otra (si no, la estría sale punteada), y la amplitud por muestra se
    corrige por `largo / muestras` para que el brillo depositado por píxel no
    dependa del largo de la estría.

    Retorna la máscara booleana de los píxeles que son humo SOLO por los
    filamentos (los que no venían ya marcados por draw_center/draw_smoke/
    draw_white_blobs, o sea: periferia fibrosa, no núcleo). trajectories.py la
    usa para decidir dónde una trayectoria se ve por encima del humo y dónde
    queda oculta detrás — ver _SMOKE_OVERRIDE_PROB. Devuelve None si no se
    pasó mask, y todo False si no se dibujó nada.
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
        return None if mask is None else np.zeros((h, w), dtype=bool)

    # bincount en índices aplanados: mucho más rápido que np.add.at con este
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

    # Periferia fibrosa = lo que marcan los filamentos y NO estaba ya marcado
    # como humo por el núcleo (draw_center/draw_smoke/draw_white_blobs). Se
    # calcula antes de escribir la máscara, que es el único momento en que los
    # dos aportes todavía se distinguen.
    filament_only = drawn & (mask != 1)
    mask[drawn] = 1
    return filament_only


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
