import numpy as np


def perlin_noise_2d(shape: tuple[int, int], scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    h, w = shape
    noise = np.zeros((h, w))

    for octave in range(octaves):
        freq = 2 ** octave
        amp = 0.5 ** octave
        grid_h = max(2, int(h / (scale / freq)))
        grid_w = max(2, int(w / (scale / freq)))

        angles = rng.uniform(0, 2 * np.pi, (grid_h + 1, grid_w + 1))
        gradients_y = np.sin(angles)
        gradients_x = np.cos(angles)

        ys = np.linspace(0, grid_h - 1, h, endpoint=False)
        xs = np.linspace(0, grid_w - 1, w, endpoint=False)

        y0 = ys.astype(int)
        x0 = xs.astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        fy = ys - y0
        fx = xs - x0

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

        d00_y = fy_col
        d00_x = fx_row
        d10_y = fy_col - 1
        d10_x = fx_row
        d01_y = fy_col
        d01_x = fx_row - 1
        d11_y = fy_col - 1
        d11_x = fx_row - 1

        g00 = gradients_y[y0_col, x0_row] * d00_y + gradients_x[y0_col, x0_row] * d00_x
        g10 = gradients_y[y1_col, x0_row] * d10_y + gradients_x[y1_col, x0_row] * d10_x
        g01 = gradients_y[y0_col, x1_row] * d01_y + gradients_x[y0_col, x1_row] * d01_x
        g11 = gradients_y[y1_col, x1_row] * d11_y + gradients_x[y1_col, x1_row] * d11_x

        lerp_x0 = g00 + fx_s * (g01 - g00)
        lerp_x1 = g10 + fx_s * (g11 - g10)
        value = lerp_x0 + fy_s * (lerp_x1 - lerp_x0)

        noise += value * amp

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise
