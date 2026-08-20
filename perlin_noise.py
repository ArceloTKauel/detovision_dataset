"""
perlin_noise.py - Ruido Perlin 2D con octavas, vectorizado.

Da la textura orgánica del humo —distorsión del borde, variación de brillo— y
decide dónde caen sus manchas sustractivas.
"""

import numpy as np


def perlin_noise_2d(shape: tuple[int, int], scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Campo de ruido fractal [shape] en [0, 1]. `scale` es el tamaño de celda en
    píxeles y `octaves` cuántas capas se suman, cada una al doble de frecuencia y
    la mitad de amplitud."""
    h, w = shape
    noise = np.zeros((h, w))

    for octave in range(octaves):
        freq = 2 ** octave      # cada octava duplica la frecuencia
        amp = 0.5 ** octave      # cada octava reduce la amplitud a la mitad

        # Grilla de gradientes para esta octava
        grid_h = max(2, int(h / (scale / freq)))
        grid_w = max(2, int(w / (scale / freq)))

        # Gradientes aleatorios en cada nodo de la grilla (vectores unitarios)
        angles = rng.uniform(0, 2 * np.pi, (grid_h + 1, grid_w + 1))
        gradients_y = np.sin(angles)
        gradients_x = np.cos(angles)

        # Mapear cada píxel a su posición dentro de la grilla
        ys = np.linspace(0, grid_h - 1, h, endpoint=False)
        xs = np.linspace(0, grid_w - 1, w, endpoint=False)

        # Índices de las celdas y posición fraccional dentro de cada una
        y0 = ys.astype(int)
        x0 = xs.astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        fy = ys - y0
        fx = xs - x0

        # Curva hermite: elimina los artefactos de grilla
        fy_smooth = fy * fy * (3 - 2 * fy)
        fx_smooth = fx * fx * (3 - 2 * fx)

        fy_s = fy_smooth[:, np.newaxis]
        fx_s = fx_smooth[np.newaxis, :]
        fy_col = fy[:, np.newaxis]
        fx_row = fx[np.newaxis, :]

        y0_col = y0[:, np.newaxis]
        x0_row = x0[np.newaxis, :]
        y1_col = y1[:, np.newaxis]
        x1_row = x1[np.newaxis, :]

        # Distancia desde cada esquina de la celda al píxel
        d00_y = fy_col
        d00_x = fx_row
        d10_y = fy_col - 1
        d10_x = fx_row
        d01_y = fy_col
        d01_x = fx_row - 1
        d11_y = fy_col - 1
        d11_x = fx_row - 1

        # Producto punto entre gradiente y vector distancia en cada esquina
        g00 = gradients_y[y0_col, x0_row] * d00_y + gradients_x[y0_col, x0_row] * d00_x
        g10 = gradients_y[y1_col, x0_row] * d10_y + gradients_x[y1_col, x0_row] * d10_x
        g01 = gradients_y[y0_col, x1_row] * d01_y + gradients_x[y0_col, x1_row] * d01_x
        g11 = gradients_y[y1_col, x1_row] * d11_y + gradients_x[y1_col, x1_row] * d11_x

        # Interpolación bilineal con los valores suavizados
        lerp_x0 = g00 + fx_s * (g01 - g00)
        lerp_x1 = g10 + fx_s * (g11 - g10)
        value = lerp_x0 + fy_s * (lerp_x1 - lerp_x0)

        noise += value * amp

    # Normalizar al rango [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise
