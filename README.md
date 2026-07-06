# Detovision Dataset Generator

Generador sintético de imágenes de explosiones, diseñado para crear datasets de entrenamiento para modelos de visión por computadora (segmentación semántica).

## Qué genera

Pares de imágenes PNG de 1280x720 generados simultáneamente en un solo pase:

- **Entrada (B/W)**: imagen binarizada en blanco y negro con trayectorias punteadas, humo con textura y manchas, y franjas de derrumbe (cuando ocurren).
- **Salida (paleta)**: máscara de segmentación en PNG modo paleta ("P") — cada píxel es un índice de clase (0-3), con paleta embebida solo para visualización:
  - **Rojo** = fondo
  - **Verde** = humo
  - **Azul** = trayectorias (líneas continuas, sin gaps)
  - **Amarillo** = derrumbe (franjas de desprendimiento de tierra/rocas)

## Estructura del proyecto

```
main.py               → Punto de entrada. Orquesta el pipeline y genera ambas imágenes.
generate_dataset.py   → Generación masiva (10k pares) con multiprocessing.
canvas.py             → Lienzo, cuadrilátero de impacto, centros de fragmentos.
smoke.py              → Humo con zonas concéntricas + manchas sustractivas.
landslide.py           → Franjas de derrumbe: bordes quebrados + textura de dientes perpendiculares.
trajectories.py       → Trayectorias rectas y parabólicas con spacing cuadrático + ráfagas.
perlin_noise.py       → Implementación de Perlin noise 2D con octavas.
export.py             → Conversión de tensor a imagen PNG (B/W) y máscara a PNG paleta.
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
   - En máscara: líneas continuas completas → azul
8. Dibujar franjas de derrumbe (~40% de las imágenes, 3-6 franjas): bordes quebrados + textura de dientes perpendiculares → amarillo en la máscara, excluyendo un radio alrededor de la explosión
9. Binarizar B/W: píxeles ≥ 128 → blanco, resto → negro

## Uso

### Prueba rápida (4 pares)

```bash
uv run main.py
```

Genera 4 pares: `explosion_N.png` (entrada B/W) + `explosion_N_mask.png` (salida RGB).

### Generación del dataset completo (10k pares)

```bash
uv run generate_dataset.py
```

Genera 10,000 pares de imágenes usando multiprocessing (autodetecta cores de la CPU).
Las imágenes se guardan en:

```
dataset/
    inputs/     → 00000.png a 09999.png (B/W binarizadas)
    targets/    → 00000.png a 09999.png (máscaras RGB)
```

Cada índice se usa como semilla aleatoria, por lo que el dataset es reproducible.

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
La máscara se construye durante el mismo pase de generación usando un tensor de etiquetas (0=fondo, 1=humo, 2=trayectoria, 3=derrumbe). La prioridad de clases es **humo > trayectoria > derrumbe > fondo**: cada clase de menor prioridad solo se marca sobre píxeles que todavía son fondo, así que nunca pisa a una clase de mayor prioridad ya dibujada. Además, el derrumbe nunca dibuja (ni en la imagen B/W ni en la máscara) dentro de un radio de exclusión alrededor del origen de la explosión, para que nunca pase por encima del humo aunque este tenga huecos (manchas sustractivas). Tras la binarización del B/W, se sincronizan los píxeles de humo que no sobrevivieron el umbral (brillo < 128) reseteándolos a fondo en la máscara. Al exportar, las etiquetas se guardan como PNG modo paleta ("P"): cada píxel es el índice de clase, y la paleta (Rojo/Verde/Azul/Amarillo) es solo metadata para visualización.
