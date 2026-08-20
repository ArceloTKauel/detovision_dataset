# Detovision Dataset Generator

Generador sintético de explosiones de voladura para entrenar segmentación
semántica. Produce imágenes de 768x512 en escala de grises (la entrada) junto con
su máscara RGB (la salida), en un solo pase que comparte todos los sorteos.

Las clases en la máscara:

| color | clase | |
|---|---|---|
| rojo | fondo | incluye el terreno, que es un artefacto de cámara y no un objeto |
| verde | humo | la pluma, sus filamentos y las sub-nubes incandescentes |
| azul | trayectoria | con gradiente real vía heatmap, no un color plano |
| amarillo | derrumbe | desactivada, ver `DRAW_LANDSLIDES` en `main.py` |

## Dos formatos

**Imagen única** — una explosión completa por par entrada/máscara.

```bash
uv run main.py               # 4 pares de prueba en la raíz
uv run generate_dataset.py   # 10.000 pares en dataset/{inputs,targets}/
```

**Secuencia temporal** — la misma explosión repartida en 90 frames, donde cada uno
muestra solo lo que nace en su tramo (no el acumulado). La unidad de entrenamiento
es un bloque de 9 frames apilados como canales, con una máscara por bloque.

```bash
uv run sequence.py                    # una secuencia de ejemplo en sequence/
uv run sequence.py 7                  # semilla fija, para comparar cambios
uv run sequence.py 7 36 acc           # 36 frames, en modo acumulado
uv run generate_sequence_dataset.py   # 3.000 secuencias en dataset_sequences/
```

El índice de cada muestra es su semilla, así que los dos datasets son
reproducibles. La propiedad que ata los dos formatos: el último frame de una
secuencia acumulada es bit a bit lo que devuelve `generate_explosion` con el mismo
`rng`.

## Mapa

```
main.py                       el pipeline de una explosión, en un solo pase
sequence.py                   la misma explosión repartida en frames, por retroceso
generate_dataset.py           generación masiva, imagen única
generate_sequence_dataset.py  generación masiva, secuencias por bloques

canvas.py        lienzo, línea de tiro (la fila de pozos) y centros de fragmento
terrain.py       manchón de parallax de terreno: la textura de fondo del dominio
smoke.py         la pluma (casco convexo + aros), filamentos y sub-nubes
trajectories.py  metralla: rectas, lazos que vuelven al origen y arcos de sobrevuelo
landslide.py     franjas de derrumbe (clase desactivada)
perlin_noise.py  ruido Perlin 2D con octavas
export.py        tensores y máscaras a PNG, y contact sheets de una secuencia
```

Cada módulo lleva arriba qué hace y por qué, y cada constante la medición sobre
las referencias reales que la fija.

## Herramientas de inspección

```bash
uv run pixel_inspector_gui.py [imagen]   # zoom, selección por clase e histogramas
uv run pixel_histogram.py <imagen> --punto X Y etiqueta ...
uv run zoom_preview.py <imagen> [--box x0 y0 x1 y1] [--zoom N]
```

## Dependencias

numpy, Pillow, tqdm (barra de avance de la generación masiva) y matplotlib, este
último solo para las herramientas de inspección.
