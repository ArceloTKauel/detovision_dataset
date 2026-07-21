# Detovision Dataset Generator

Generador sintético de imágenes de explosiones, diseñado para crear datasets de entrenamiento para modelos de visión por computadora (segmentación semántica).

## Qué genera

Pares de imágenes PNG de 768x512 (ancho x alto) generados simultáneamente en un solo pase:

- **Entrada (escala de grises)**: imagen en escala de grises continua (no binarizada) con humo texturizado (+ sub-nubes de humo blanco), trayectorias punteadas y franjas de derrumbe (cuando ocurren; clase actualmente desactivada, ver `DRAW_LANDSLIDES` en `main.py`).
- **Salida (RGB)**: máscara de segmentación en PNG modo RGB — fondo/humo/derrumbe con color plano, trayectoria con gradiente real vía heatmap:
  - **Rojo** = fondo
  - **Verde** = humo
  - **Azul** (gradiente: intenso al centro, tenue hacia el borde) = trayectorias
  - **Amarillo** = derrumbe (franjas de desprendimiento de tierra/rocas)

## Estructura del proyecto

```
main.py               → Punto de entrada. Orquesta el pipeline y genera ambas imágenes.
generate_dataset.py   → Generación masiva (10k pares) con multiprocessing.
canvas.py             → Lienzo, cuadrilátero de impacto, centros de fragmentos.
smoke.py              → Humo con zonas concéntricas + manchas sustractivas + sub-nubes de humo blanco.
landslide.py           → Franjas de derrumbe: bordes quebrados + textura de dientes perpendiculares.
trajectories.py       → Trayectorias rectas, parabólicas en lazo y de sobrevuelo, con spacing cuadrático + ráfagas.
perlin_noise.py       → Implementación de Perlin noise 2D con octavas.
export.py             → Conversión de tensor a imagen PNG en escala de grises y máscara a PNG RGB.
```

## Pipeline de generación

Ambas salidas (escala de grises y máscara RGB) se generan en el mismo pase, compartiendo todos los cálculos y parámetros aleatorios:

1. Crear lienzo vacío (512 alto x 768 ancho, fondo negro) + máscara de segmentación + heatmap de gradiente para trayectorias
2. Generar cuadrilátero aleatorio → zona de impacto
3. Calcular centroide → origen de la explosión
4. Sortear la escala de brillo global y dibujar el centro principal + 40-80 centros secundarios dentro del cuadrilátero, camuflados por debajo del brillo del núcleo de humo (marcados como humo en la máscara)
5. Dibujar humo con Perlin noise (4 zonas + manchas sustractivas → verde en la máscara), más 1-2 sub-nubes de "humo blanco" (70% de probabilidad) simulando metralla/brasas incandescentes
6. Dibujar trayectorias de metralla: 15-30 rectas, 15-30 parabólicas en lazo (vuelven al punto de partida) y 1-4 de sobrevuelo (arco abierto que pasa por encima de la nube y aterriza en otro punto):
   - En la entrada: punteadas con spacing cuadrático + ráfagas de 1-5 píxeles
   - En la máscara: azul con gradiente real vía heatmap (no líneas planas)
7. Dibujar franjas de derrumbe (actualmente desactivado, `DRAW_LANDSLIDES = False`; cuando está activo: ~40% de las imágenes, 1-8 franjas): bordes quebrados + textura de dientes perpendiculares → amarillo en la máscara, excluyendo un radio alrededor de la explosión
8. Sincronizar máscara: píxeles de humo que quedaron en negro puro (p. ej. manchas sustractivas que no dibujaron nada) vuelven a fondo

## Uso

### Prueba rápida (4 pares)

```bash
uv run main.py
```

Genera 4 pares: `explosion_N.png` (entrada en escala de grises) + `explosion_N_mask.png` (salida RGB).

### Generación del dataset completo (10k pares)

```bash
uv run generate_dataset.py
```

Genera 10,000 pares de imágenes usando multiprocessing (autodetecta cores de la CPU).
Las imágenes se guardan en:

```
dataset/
    inputs/     → 00000.png a 09999.png (escala de grises, sin binarizar)
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
Se usan cuatro capas de Perlin noise:
- **Grueso** (scale grande, 5 octavas): distorsiona los bordes del humo para que no sean circulares.
- **Fino** (scale pequeño, 3 octavas): varía el brillo dentro del core.
- **Grano** (scale muy pequeña, 2 octavas): rompe la superficie lisa con textura fibrosa/granulada, incluso dentro del core.
- **Sustractivo** (3 octavas): determina dónde se generan las manchas/huecos.

### Sub-nubes de humo blanco
`draw_white_blobs` no dibuja puntos sueltos: elige un centro al azar dentro del humo ya trazado y vuelve a llamar a `draw_smoke` con un radio más chico (35-60% de `smoke_radius`) y un piso de brillo alto (130 en vez de 15). Así la sub-nube hereda la misma textura orgánica (zonas core/mid/outer/fringe + Perlin) que el humo normal, pero se lee como un núcleo incandescente en vez de gris — simula metralla o brasas agrupadas dentro de la nube.

### Tipos de trayectoria
Las tres variantes rasterizan con Bresenham + spacing cuadrático + ráfagas, pero difieren en geometría: **rectas** (`draw_trajectory`), **parabólicas en lazo** (`draw_returning_parabola`, describen una elipse completa y cierran exactamente sobre su punto de partida) y **de sobrevuelo** (`draw_flyover_trajectory`, medio lazo que se eleva por encima de la nube de humo y aterriza en otro punto, sin volver al origen).

### Máscara de segmentación
La máscara se construye durante el mismo pase de generación usando un tensor de etiquetas (0=fondo, 1=humo, 2=trayectoria, 3=derrumbe) más un heatmap aparte para el gradiente de trayectoria. La prioridad de clases es **humo > trayectoria > derrumbe > fondo**: cada clase de menor prioridad solo se marca sobre píxeles que todavía son fondo, así que nunca pisa a una clase de mayor prioridad ya dibujada. Además, el derrumbe nunca dibuja (ni en la entrada ni en la máscara) dentro de un radio de exclusión alrededor del origen de la explosión, para que nunca pase por encima del humo aunque este tenga huecos (manchas sustractivas). Al final del pase se sincronizan los píxeles de humo que quedaron en negro puro (p. ej. manchas sustractivas que no dibujaron nada) reseteándolos a fondo en la máscara. Al exportar (`export.py::mask_to_rgb`), fondo/humo/derrumbe se pintan con color plano (rojo/verde/amarillo) y la trayectoria usa el canal azul con la intensidad del heatmap — no es PNG modo paleta, porque una paleta de 256 entradas no puede representar el gradiente continuo.
