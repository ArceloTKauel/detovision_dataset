# Detovision Dataset Generator

Generador sintético de imágenes de explosiones en escala de grises, diseñado para crear datasets de entrenamiento para modelos de visión por computadora.

## Qué genera

Imágenes PNG de 1280x720 en blanco y negro que simulan explosiones vistas desde un dron. Cada imagen incluye:

- **Zona de impacto**: cuadrilátero aleatorio con centros de fragmentos distribuidos uniformemente.
- **Humo**: nube orgánica generada con Perlin noise, dividida en zonas concéntricas (core, mid, outer, fringe) con manchas sustractivas (huecos).
- **Trayectorias de metralla**: líneas punteadas rectas y parabólicas que salen de la explosión, con spacing variable (denso cerca, disperso lejos).

## Estructura del proyecto

```
main.py             → Punto de entrada. Orquesta el pipeline y genera las imágenes.
canvas.py           → Lienzo, cuadrilátero de impacto, centros de fragmentos.
smoke.py            → Humo con zonas concéntricas + manchas sustractivas.
trajectories.py     → Trayectorias rectas y parabólicas con spacing cuadrático.
perlin_noise.py     → Implementación de Perlin noise 2D con octavas.
export.py           → Conversión de tensor numpy a imagen PNG.
```

## Pipeline de generación

1. Crear lienzo vacío (720x1280, fondo negro)
2. Generar cuadrilátero aleatorio → zona de impacto
3. Calcular centroide → origen de la explosión
4. Distribuir 40-80 centros dentro del cuadrilátero
5. Dibujar humo con Perlin noise (4 zonas + manchas sustractivas)
6. Calcular ángulo del dron (influye en curvatura de parábolas)
7. Dibujar trayectorias rectas (1-15) y parabólicas (1-15)
8. Binarizar: píxeles ≥ 128 → blanco, resto → negro

## Uso

```bash
uv run main.py
```

Genera 4 imágenes: `explosion_1.png` a `explosion_4.png`.

## Dependencias

- **numpy**: operaciones matriciales y generación aleatoria
- **Pillow (PIL)**: exportación a PNG y dibujo de polígonos sustractivos

## Conceptos clave

### Spacing cuadrático de trayectorias
Los puntos de las trayectorias no son equidistantes. El spacing sigue `ratio² × max_spacing`, donde `ratio` es la distancia al origen normalizada (0 a 1). Esto hace que los puntos sean densos cerca de la explosión y dispersos lejos, simulando la desaceleración de la metralla.

### Perlin noise en el humo
Se usan dos capas de Perlin noise:
- **Grueso** (scale grande, 5 octavas): distorsiona los bordes del humo para que no sean circulares.
- **Fino** (scale pequeño, 3 octavas): varía el brillo dentro del core.
- **Sustractivo** (3 octavas): determina dónde se generan las manchas/huecos.

### Influencia del ángulo del dron
El dron "mira" hacia el centro del lienzo. Este ángulo modula la curvatura de las trayectorias parabólicas: trayectorias perpendiculares al dron curvan más (efecto de perspectiva), las paralelas casi nada.
