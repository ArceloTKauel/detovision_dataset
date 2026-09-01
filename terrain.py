"""
terrain.py - Campo de relieve del fondo (ego-motion de cámara).

La textura de fondo que aparece en video real al diferenciar frames consecutivos,
por el movimiento de la cámara sobre el relieve. El modelo la confundía con la
clase trayectoria (ver mascara_cambios_final_sinbin_*.png en el repo hermano).

Se dibuja primero y no toca `mask`: queda como fondo salvo donde otro elemento lo
cubra, que es lo correcto — es un artefacto de cámara, no un objeto a segmentar.

Lo que este módulo produce es el RELIEVE, no el residuo. El residuo lo arma
sequence.py desplazando este campo entre frames, y es proporcional a su GRADIENTE:
por eso acá lo que importa es la estructura de bordes, no el brillo.

Medido sobre 4 videos reales (2026-08-28), el residuo de un frame crece con la
separación temporal siguiendo |∇campo|·|desplazamiento| — a 40 frames de distancia
la predicción acierta con 2% de error. Confirma que el mecanismo es geométrico.

El generador es el que entregó el equipo (TerrainDemo.ipynb), portado a numpy+PIL
y adaptado; las tres diferencias están anotadas en cada función.
"""

import numpy as np
from PIL import Image, ImageFilter

# En las siete referencias el terreno llena el cuadro desde el primer frame, así
# que un frame sin terreno no es un caso difícil sino una entrada sin información.
TERRAIN_PROB = 1.0                           # probabilidad de que la imagen lleve terreno

# ── Ruido fractal base ─────────────────────────────────────────────────────
# La octava 0 es la de mayor resolución y la que menos pesa; el tope de amplitud
# crece con el índice, así que el relieve lo dominan las frecuencias bajas y las
# altas solo lo texturan.
TERRAIN_OCTAVES     = 8                      # octavas del ruido fractal
TERRAIN_OCTAVE_GAIN = 0.1                    # tope de amplitud de la octava i: (i+1)*gain

# ── Dirección dominante ────────────────────────────────────────────────────
# Un banco real está cortado en bermas paralelas, así que su relieve tiene
# lineamientos largos en una dirección. Medido sobre 9 frames quietos de los 8
# videos (razón entre autovalores de la matriz de estructura, sobre bloques de
# 8x8): el real va de 1.28 a 2.60 y el ruido fractal isótropo daba 1.00-1.30 —
# era la única de las ocho varas fuera de rango.
#
# Se estira la grilla de cada octava en una sola dirección y después se rota el
# campo entero a un ángulo sorteado. El estirado va en la grilla y no en un
# resize posterior porque estirar después agranda TODAS las escalas por igual y
# el grano fino se pierde; estirando la grilla, solo se alargan las estructuras.
#
# El valor va MUY por encima de lo que se lee en el residuo, y no es un descuido:
# el residuo es |∇campo·desplazamiento|, que rectifica la dirección, y encima el
# piso de ruido es isótropo. Medido, un campo con anisotropía 4.5 deja un residuo
# de 1.0-1.8. Barrido de 1 a 30 (2026-08-31): la respuesta es monótona y con este
# rango el residuo da 1.55, contra 1.28-2.60 del real. La otra palanca es bajar
# TERRAIN_WARP_STRENGTH —a (1, 8) sube a 1.82— pero eso devuelve el campo a
# manchas isótropas, así que se prefiere alargar más y no distorsionar menos.
TERRAIN_ELONGATION = (4.0, 10.0)             # cuánto se alarga el relieve en su dirección

# ── Distorsión del dominio ─────────────────────────────────────────────────
# El ingrediente que convierte ruido fractal en relieve: cada píxel se lee de otra
# posición, desplazada por un segundo campo fractal. Sin esto el campo son manchas
# isotrópicas; con esto aparecen las estructuras alargadas y marmoladas que tienen
# los bancos reales.
TERRAIN_WARP_STRENGTH = (1.0, 64.0)          # desplazamiento máximo del dominio, en px

# ── Curva de transferencia ─────────────────────────────────────────────────
# El relieve pasa por una curva ALEATORIA Y NO MONÓTONA antes de ser brillo. Es lo
# que produce las bandas: donde la curva es empinada el residuo se enciende, donde
# es plana se apaga, y qué banda se enciende cambia con la dirección del movimiento.
#
# El notebook lo hacía armando un colormap RGB en HSL y pasando a gris después; el
# rodeo por RGB existe porque estaba escrito para terrenos en color. En gris el
# efecto neto es exactamente este: puntos de anclaje sorteados en posición y en
# valor, interpolados. La no-monotonía venía de que dos colores de la misma
# luminancia dan grises muy distintos (azul puro 0.11, amarillo puro 0.89).
#
# La interpolación es SUAVE y no escalonada. El notebook sorteaba entre las dos,
# pero una curva constante a tramos hace el campo constante a tramos, y una meseta
# plana tiene gradiente cero: no produce residuo. Medido, la escalonada deja muerto
# el 80% del cuadro con 4 anclajes y todavía el 37% con 20, contra el 15% de ceros
# del real; la suave con 6 anclajes deja 1.5%. Por eso el mínimo es 6 y no 2.
TERRAIN_CURVE_ANCHORS = (6, 14)              # puntos de anclaje de la curva

# ── Brillo ─────────────────────────────────────────────────────────────────
# Bajado de (15, 55) el 2026-08-31: alargar el relieve subió el brillo del campo un
# 60% por sí solo (media 6.97 -> 11.13), y con él el residuo se fue por encima del
# real (media hasta 1.9 contra 1.37). El factor 0.65 sale de un barrido de 5 puntos:
# es donde la media vuelve al rango real perdiendo lo menos posible de anisotropía,
# que se apaga al bajar el nivel porque el piso de ruido isótropo pesa más.
TERRAIN_BLOTCH_MAX_VAL  = (10, 36)           # brillo máximo del campo, antes de intensity
TERRAIN_INTENSITY_RANGE = (0.05, 1.0)        # brillo global: varía 20x entre imágenes

# El blur no es cosmético: define el ANCHO del residuo, porque desplazar un borde
# duro devuelve una línea de un píxel y desplazar uno blando devuelve una franja.
# Con (0.3, 0.8) el grano espacial del residuo daba 0.43 contra 0.53-0.74 real y no
# subía tocando el ruido — se saturaba en 0.45. Con (1.2, 2.0) da 0.62. El precio
# es que ablanda el campo y con él baja el residuo, que se compensa subiendo
# TERRAIN_STEP_RANGE en sequence.py: las dos constantes van juntas.
TERRAIN_BLUR_RADIUS     = (1.2, 2.0)         # suavizado final, en px


def _stretch_to_unit_range(field):
    field = field - field.min()
    return field / (field.max() + 1e-8)


def _fractal_noise(h, w, channels, rng, elongation=1.0):
    """Campo [h, w, channels] de ruido fractal gaussiano.

    Sobre `rng` y no sobre el estado global de numpy, para que la generación quede
    determinada por la semilla de generate_dataset.py.

    El notebook estira cada octava con `cv2.resize(x, (h, w))`, que pasa los
    argumentos al revés de lo que cv2 espera; no se nota porque ahí las imágenes
    son cuadradas. Acá el lienzo es 768x512 y se resuelve bien.

    `elongation` adelgaza la grilla en x, así que al estirarla a [h, w] las
    estructuras salen alargadas en esa dirección (ver TERRAIN_ELONGATION).
    """
    field = np.zeros((h, w, channels), dtype=np.float32)
    for octave in range(TERRAIN_OCTAVES):
        amplitude = rng.uniform(0.0, (octave + 1) * TERRAIN_OCTAVE_GAIN)
        oh = max(2, h // (2 ** octave))
        ow = max(2, int(w // (2 ** octave * elongation)))
        layer = rng.normal(0.0, 1.0, size=(oh, ow, channels)).astype(np.float32)
        if (oh, ow) != (h, w):
            layer = np.stack([
                np.asarray(Image.fromarray(layer[..., c], mode="F").resize((w, h), Image.BILINEAR))
                for c in range(channels)], axis=-1)
        field += amplitude * layer
    return field


def _domain_warp(field, rng):
    """Lee cada píxel desde otra posición, desplazada por un campo fractal aparte.

    El muestreo es por vecino más cercano, como en el notebook: deja escalones de
    un píxel que son parte de la textura y que el blur final atenúa."""
    h, w = field.shape
    delta = _fractal_noise(h, w, 2, rng)
    delta = (_stretch_to_unit_range(delta) * 2.0 - 1.0) * rng.uniform(*TERRAIN_WARP_STRENGTH)

    yy, xx = np.mgrid[0:h, 0:w]
    x2 = np.clip(xx + delta[..., 0], 0, w - 1).astype(np.int32)
    y2 = np.clip(yy + delta[..., 1], 0, h - 1).astype(np.int32)
    return field[y2, x2]


def _transfer_curve(rng):
    """Curva [0, 1] -> [0, 1] de 256 entradas, aleatoria y no monótona.

    Se normaliza al rango completo: con anclajes libres, un sorteo desafortunado da
    una curva casi plana y el campo sale sin contraste — y el contraste ES la señal,
    porque el residuo va con el gradiente. Sin esto el gradiente medio variaba 90x
    entre muestras."""
    n = int(rng.integers(*TERRAIN_CURVE_ANCHORS, endpoint=True))
    positions = np.sort(rng.uniform(0.0, 1.0, n))
    values = rng.uniform(0.0, 1.0, n)
    curve = np.interp(np.linspace(0.0, 1.0, 256), positions, values)
    return _stretch_to_unit_range(curve.astype(np.float32))


def _oriented_relief(h, w, rng):
    """Relieve con una dirección dominante sorteada: ruido alargado en x y rotado.

    Se genera sobre un cuadrado del tamaño de la diagonal y se recorta el centro,
    porque rotar el lienzo justo dejaría las esquinas vacías. El ángulo va en
    [0, 180): un lineamiento no tiene sentido, girarlo media vuelta da el mismo.
    """
    side = int(np.ceil(np.hypot(h, w)))
    field = _fractal_noise(side, side, 1, rng, rng.uniform(*TERRAIN_ELONGATION))[..., 0]

    rotated = Image.fromarray(field, mode="F").rotate(
        rng.uniform(0.0, 180.0), Image.BILINEAR)
    top, left = (side - h) // 2, (side - w) // 2
    return np.asarray(rotated, dtype=np.float32)[top:top + h, left:left + w]


def terrain_field(h, w, rng):
    """Campo de relieve [h, w] en [0, 1]: ruido fractal orientado, distorsionado
    en el dominio y pasado por la curva de transferencia.

    La distorsión va DESPUÉS de orientar: aplicada antes, la rotación arrastraría
    un campo ya deformado y los lineamientos saldrían más rectos de lo que son.
    """
    relief = _stretch_to_unit_range(_oriented_relief(h, w, rng))
    relief = _domain_warp(relief, rng)
    curve = _transfer_curve(rng)
    return curve[np.clip(relief * 255.0, 0, 255).astype(np.int32)]


def _terrain_brightness(h, w, rng):
    """Mapa de brillo [0, 255] del campo, escalado por la intensidad global."""
    field = terrain_field(h, w, rng)

    intensity = rng.uniform(*TERRAIN_INTENSITY_RANGE)
    max_val = rng.uniform(*TERRAIN_BLOTCH_MAX_VAL) * intensity
    image = Image.fromarray((field * max_val).astype(np.uint8))

    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*TERRAIN_BLUR_RADIUS)))
    return np.array(image, dtype=np.float32)


def draw_terrain(
    tensor: np.ndarray,
    rng: np.random.Generator,
    prob: float = TERRAIN_PROB,
) -> None:
    """Sortea si esta imagen lleva terreno y lo dibuja con np.maximum, el mismo
    criterio de composición que el resto del pipeline."""
    if rng.random() >= prob:
        return

    h, w = tensor.shape
    brightness = _terrain_brightness(h, w, rng)
    tensor[:] = np.maximum(tensor, brightness.astype(np.uint8))
