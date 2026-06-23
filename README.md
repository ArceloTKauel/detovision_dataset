# Detovision Dataset Generator

Generador sintético de imágenes de explosiones, diseñado para crear datasets de entrenamiento para modelos de visión por computadora (segmentación semántica).

## Qué genera

Pares de imágenes PNG de 1280x720 generados simultáneamente en un solo pase:

- **Entrada (B/W)**: imagen binarizada en blanco y negro con trayectorias punteadas, humo con textura y manchas.
- **Salida (RGB)**: máscara de segmentación con los mismos elementos coloreados por clase:
  - **Rojo** = trayectorias (líneas continuas, sin gaps)
  - **Verde** = humo
  - **Azul** = fondo

## Estructura del proyecto

```
main.py             → Punto de entrada. Orquesta el pipeline y genera ambas imágenes.
canvas.py           → Lienzo, cuadrilátero de impacto, centros de fragmentos.
smoke.py            → Humo con zonas concéntricas + manchas sustractivas.
trajectories.py     → Trayectorias rectas y parabólicas con spacing cuadrático + ráfagas.
perlin_noise.py     → Implementación de Perlin noise 2D con octavas.
export.py           → Conversión de tensor a imagen PNG (B/W y RGB).
```

## Pipeline de generación

Ambas salidas (B/W y máscara RGB) se generan en el mismo pase, compartiendo todos los cálculos y parámetros aleatorios:

1. Crear lienzo vacío (720x1280, fondo negro) + máscara de segmentación
2. Generar cuadrilátero aleatorio → zona de impacto
3. Calcular centroide → origen de la explosión
4. Distribuir 40-80 centros dentro del cuadrilátero (marcados como humo en la máscara)
5. Dibujar humo con Perlin noise (4 zonas + manchas sustractivas → verde en la máscara)
6. Calcular ángulo del dron (influye en curvatura de parábolas)
7. Dibujar trayectorias rectas (1-15) y parabólicas (1-15):
   - En B/W: punteadas con spacing cuadrático + ráfagas de 1-5 píxeles
   - En máscara: líneas continuas completas → rojo
8. Binarizar B/W: píxeles ≥ 128 → blanco, resto → negro

## Uso

```bash
uv run main.py
```

Genera 4 pares: `explosion_N.png` (entrada B/W) + `explosion_N_mask.png` (salida RGB).

## Dependencias

- **numpy**: operaciones matriciales y generación aleatoria
- **Pillow (PIL)**: exportación a PNG y dibujo de polígonos sustractivos

## Conceptos clave

### Spacing cuadrático + ráfagas en trayectorias
Los puntos de las trayectorias no son equidistantes. El spacing sigue `ratio² × max_spacing`, donde `ratio` es la distancia al origen normalizada (0 a 1). Esto hace que los puntos sean densos cerca de la explosión y dispersos lejos. Además, cada punto inicia una ráfaga de 1-5 píxeles consecutivos (70% de probabilidad cada uno), generando agrupaciones orgánicas (`. ...  .. .....  .`) en vez de puntos solitarios (`. . . . .`).

### Perlin noise en el humo
Se usan tres capas de Perlin noise:
- **Grueso** (scale grande, 5 octavas): distorsiona los bordes del humo para que no sean circulares.
- **Fino** (scale pequeño, 3 octavas): varía el brillo dentro del core.
- **Sustractivo** (3 octavas): determina dónde se generan las manchas/huecos.

### Influencia del ángulo del dron
El dron "mira" hacia el centro del lienzo. Este ángulo modula la curvatura de las trayectorias parabólicas: trayectorias perpendiculares al dron curvan más (efecto de perspectiva), las paralelas casi nada.

### Máscara de segmentación
La máscara se construye durante el mismo pase de generación usando un tensor de etiquetas (0=fondo, 1=humo, 2=trayectoria). Las trayectorias tienen prioridad sobre el humo: si una trayectoria cruza el humo, ese píxel se marca como trayectoria (rojo). Al exportar, las etiquetas se convierten a RGB (Azul/Verde/Rojo).
