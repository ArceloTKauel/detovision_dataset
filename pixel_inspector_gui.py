"""
pixel_inspector_gui.py - Inspector interactivo de píxeles, tipo Photoshop:
zoom hasta ver el píxel real, desplazamiento (pan) por la imagen, selección
de segmentos a mano (rectángulo) con etiqueta, y comparación de histogramas
por etiqueta (las selecciones se acumulan, igual que pixel_histogram.py pero
interactivo en vez de pasar coordenadas por CLI).

Controles:
    - Rueda del mouse: zoom in/out centrado en el cursor.
    - Click izquierdo + arrastrar: selecciona un rectángulo (pide etiqueta al soltar).
    - Click derecho + arrastrar: desplaza la vista (pan).
    - Botón "Ajustar a ventana": recentra y reescala para ver la imagen completa.
    - Panel derecho: lupa (vecindario ampliado con grilla y valores bajo el
      cursor, o de la selección mientras se arrastra) + lista de selecciones
      (por etiqueta) + histograma superpuesto.

Uso:
    uv run pixel_inspector_gui.py [imagen]

Si no se pasa imagen, se abre un diálogo para elegir el archivo.
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

MIN_ZOOM = 0.05
MAX_ZOOM = 64.0
ZOOM_STEP = 1.25

# Lupa: al mover el mouse sin arrastrar, se previsualiza un vecindario
# HOVER_KERNEL x HOVER_KERNEL centrado en el cursor, para apuntar con
# precisión antes de hacer la selección real.
HOVER_KERNEL = 11
PREVIEW_CANVAS_SIZE = 220
PREVIEW_MAX_CELL_ZOOM = 40
PREVIEW_VALUE_TEXT_MAX_CELLS = 400  # con más celdas que esto, ya no entran los números


class Selection:
    def __init__(self, label: str, x0: int, y0: int, x1: int, y1: int, values: np.ndarray, rect_id: int):
        self.label = label
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.values = values
        self.rect_id = rect_id


class PixelInspectorApp:
    def __init__(self, root: tk.Tk, image_path: str):
        self.root = root
        self.root.title(f"Pixel Inspector — {image_path}")

        self.pil_img = Image.open(image_path).convert("L")
        self.img_array = np.array(self.pil_img)
        self.img_w, self.img_h = self.pil_img.size

        self.zoom = 1.0
        self.view_x = 0.0  # esquina superior izquierda visible, en coordenadas de imagen
        self.view_y = 0.0
        self._initial_fit_done = False
        self._photo = None  # referencia viva a la imagen renderizada (evita garbage collection)
        self._preview_photo = None  # ídem, para la lupa

        self.selections: list[Selection] = []
        self.label_colors: dict[str, str] = {}
        self._drag_start = None
        self._drag_rect_id = None
        self._pan_start = None

        self._build_ui()
        self._bind_events()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Ajustar a ventana", command=self.fit_to_window).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Zoom 1:1", command=self.reset_zoom_1to1).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Eliminar selección", command=self._delete_selected).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Borrar todas", command=self._clear_selections).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Guardar histograma", command=self._save_histogram).pack(side=tk.LEFT, padx=4, pady=4)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Canvas de la imagen (izquierda) ---
        canvas_frame = ttk.Frame(main)
        self.canvas = tk.Canvas(canvas_frame, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        main.add(canvas_frame, weight=3)

        # --- Panel lateral (derecha) ---
        side = ttk.Frame(main)
        main.add(side, weight=2)

        ttk.Label(side, text="Vista previa (lupa)", font=("", 10, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))
        self.preview_canvas = tk.Canvas(
            side, width=PREVIEW_CANVAS_SIZE, height=PREVIEW_CANVAS_SIZE, bg="#101010", highlightthickness=1,
            highlightbackground="#444444",
        )
        self.preview_canvas.pack(padx=4, pady=4)

        ttk.Label(side, text="Selecciones", font=("", 10, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))
        self.tree = ttk.Treeview(side, columns=("n", "media", "std", "min", "max"), show="tree headings", height=10)
        self.tree.heading("#0", text="Etiqueta / región")
        self.tree.heading("n", text="n")
        self.tree.heading("media", text="media")
        self.tree.heading("std", text="std")
        self.tree.heading("min", text="min")
        self.tree.heading("max", text="max")
        for col, w in (("#0", 170), ("n", 55), ("media", 60), ("std", 55), ("min", 45), ("max", 45)):
            self.tree.column(col, width=w, anchor=tk.CENTER if col != "#0" else tk.W)
        self.tree.pack(fill=tk.X, padx=4, pady=4)

        fig = Figure(figsize=(4, 3.2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(fig, master=side)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._redraw_histogram()

        self.status = tk.StringVar(value="Rueda: zoom · click-izq arrastrar: seleccionar · click-der arrastrar: pan")
        ttk.Label(self.root, textvariable=self.status, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_wheel)       # Windows
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(e, delta=120))   # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(e, delta=-120))  # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self._on_select_start)
        self.canvas.bind("<B1-Motion>", self._on_select_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_select_end)

        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)

        self.canvas.bind("<Motion>", self._on_motion)

    # ------------------------------------------------------------ Vista/zoom

    def _on_configure(self, _event) -> None:
        if not self._initial_fit_done:
            self._initial_fit_done = True
            self.fit_to_window()
        else:
            self._render()

    def fit_to_window(self) -> None:
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self.zoom = max(MIN_ZOOM, min(cw / self.img_w, ch / self.img_h))
        self.view_x = (self.img_w - cw / self.zoom) / 2
        self.view_y = (self.img_h - ch / self.zoom) / 2
        self._render()

    def reset_zoom_1to1(self) -> None:
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        cx, cy = self.view_x + cw / (2 * self.zoom), self.view_y + ch / (2 * self.zoom)
        self.zoom = 1.0
        self.view_x = cx - cw / (2 * self.zoom)
        self.view_y = cy - ch / (2 * self.zoom)
        self._render()

    def _on_wheel(self, event, delta: int | None = None) -> None:
        d = event.delta if delta is None else delta
        factor = ZOOM_STEP if d > 0 else (1 / ZOOM_STEP)
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        img_x = self.view_x + event.x / self.zoom
        img_y = self.view_y + event.y / self.zoom
        self.zoom = new_zoom
        self.view_x = img_x - event.x / self.zoom
        self.view_y = img_y - event.y / self.zoom
        self._render()

    def _on_pan_start(self, event) -> None:
        self._pan_start = (event.x, event.y, self.view_x, self.view_y)

    def _on_pan_drag(self, event) -> None:
        if self._pan_start is None:
            return
        sx, sy, ovx, ovy = self._pan_start
        self.view_x = ovx - (event.x - sx) / self.zoom
        self.view_y = ovy - (event.y - sy) / self.zoom
        self._render()

    # --------------------------------------------------------------- Render

    def _render(self) -> None:
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        x0 = max(0, int(math.floor(self.view_x)))
        y0 = max(0, int(math.floor(self.view_y)))
        x1 = min(self.img_w, int(math.ceil(self.view_x + cw / self.zoom)) + 1)
        y1 = min(self.img_h, int(math.ceil(self.view_y + ch / self.zoom)) + 1)

        self.canvas.delete("img")
        self.canvas.delete("rect")

        if x1 > x0 and y1 > y0:
            crop = self.pil_img.crop((x0, y0, x1, y1))
            disp_w = max(1, round((x1 - x0) * self.zoom))
            disp_h = max(1, round((y1 - y0) * self.zoom))
            resized = crop.resize((disp_w, disp_h), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(resized)
            canvas_pos_x = (x0 - self.view_x) * self.zoom
            canvas_pos_y = (y0 - self.view_y) * self.zoom
            self.canvas.create_image(canvas_pos_x, canvas_pos_y, anchor=tk.NW, image=self._photo, tags="img")

        for sel in self.selections:
            self._draw_selection_rect(sel)

        self.canvas.tag_lower("img")

    def _image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return (ix - self.view_x) * self.zoom, (iy - self.view_y) * self.zoom

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return self.view_x + cx / self.zoom, self.view_y + cy / self.zoom

    def _draw_selection_rect(self, sel: Selection) -> None:
        color = self.label_colors[sel.label]
        cx0, cy0 = self._image_to_canvas(sel.x0, sel.y0)
        cx1, cy1 = self._image_to_canvas(sel.x1, sel.y1)
        self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline=color, width=2, tags=(f"rect_{id(sel)}", "rect"))

    # ----------------------------------------------------------- Selección

    def _on_select_start(self, event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="white", dash=(4, 2), width=1, tags="dragrect"
        )

    def _snapped_rect(self, cx0: float, cy0: float, cx1: float, cy1: float) -> tuple[int, int, int, int]:
        """Convierte dos esquinas en coordenadas de canvas a un rectángulo de
        imagen alineado a la grilla de píxeles. Usa floor(), no round(): el
        render dibuja el píxel i ocupando el rango [i, i+1) en coordenadas de
        imagen, así que floor() es lo que identifica correctamente en qué
        celda cae un click. round() hacía que un click en la mitad derecha/
        inferior del cuadradito visual de un píxel "saltara" al vecino de al
        lado, que es justo el bug reportado (a veces selecciona el de la
        derecha, o el de arriba)."""
        ix0, iy0 = self._canvas_to_image(cx0, cy0)
        ix1, iy1 = self._canvas_to_image(cx1, cy1)
        px0, px1 = sorted((math.floor(ix0), math.floor(ix1)))
        py0, py1 = sorted((math.floor(iy0), math.floor(iy1)))
        # +1: rango exclusivo para slicing, incluyendo el píxel donde cae el punto final
        x0, x1 = px0, px1 + 1
        y0, y1 = py0, py1 + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.img_w, x1), min(self.img_h, y1)
        return x0, y0, x1, y1

    def _on_select_drag(self, event) -> None:
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        x0, y0, x1, y1 = self._snapped_rect(sx, sy, event.x, event.y)
        # Redibuja el rectángulo punteado ya snapeado a la grilla de píxeles,
        # para que lo que se ve en pantalla sea exactamente lo que se va a capturar.
        cx0, cy0 = self._image_to_canvas(x0, y0)
        cx1, cy1 = self._image_to_canvas(x1, y1)
        self.canvas.coords(self._drag_rect_id, cx0, cy0, cx1, cy1)
        self.status.set(f"Seleccionando: {x1 - x0}x{y1 - y0} px")
        self._update_preview(x0, y0, x1, y1)

    def _on_select_end(self, event) -> None:
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        x0, y0, x1, y1 = self._snapped_rect(sx, sy, event.x, event.y)
        self.canvas.delete("dragrect")
        self._drag_start = None
        if x1 <= x0 or y1 <= y0:
            return
        self._update_preview(x0, y0, x1, y1)

        existing_labels = sorted(self.label_colors.keys())
        suggestion = existing_labels[-1] if existing_labels else "seleccion_1"
        label = simpledialog.askstring(
            "Etiqueta de la selección",
            f"Región ({x0},{y0})-({x1},{y1}), {x1 - x0}x{y1 - y0} px.\nNombre de clase (ej. trayectoria, fondo):",
            initialvalue=suggestion,
            parent=self.root,
        )
        if not label:
            self.status.set("Selección descartada (sin etiqueta).")
            return

        if label not in self.label_colors:
            self.label_colors[label] = PALETTE[len(self.label_colors) % len(PALETTE)]

        values = self.img_array[y0:y1, x0:x1].flatten()
        sel = Selection(label, x0, y0, x1, y1, values, rect_id=0)
        self.selections.append(sel)
        self._draw_selection_rect(sel)
        self._refresh_tree()
        self._redraw_histogram()
        self.status.set(f"Agregada selección '{label}': n={values.size}, media={values.mean():.2f}")

    def _on_motion(self, event) -> None:
        ix, iy = self._canvas_to_image(event.x, event.y)
        ix_i, iy_i = math.floor(ix), math.floor(iy)
        if 0 <= ix_i < self.img_w and 0 <= iy_i < self.img_h:
            val = int(self.img_array[iy_i, ix_i])
            self.status.set(f"img=({ix_i},{iy_i})  valor={val}  zoom={self.zoom * 100:.0f}%")
            # Si no hay un drag de selección en curso, la lupa sigue al cursor
            # (vecindario HOVER_KERNEL x HOVER_KERNEL) para apuntar con precisión.
            if self._drag_start is None:
                half = HOVER_KERNEL // 2
                hx0, hy0 = max(0, ix_i - half), max(0, iy_i - half)
                hx1, hy1 = min(self.img_w, ix_i + half + 1), min(self.img_h, iy_i + half + 1)
                self._update_preview(hx0, hy0, hx1, hy1, highlight=(ix_i, iy_i))
        else:
            self.status.set(f"zoom={self.zoom * 100:.0f}%")

    # ------------------------------------------------------------- Lupa

    def _update_preview(self, x0: int, y0: int, x1: int, y1: int, highlight: tuple[int, int] | None = None) -> None:
        self.preview_canvas.delete("all")
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return

        cell_zoom = max(1, min(PREVIEW_MAX_CELL_ZOOM, PREVIEW_CANVAS_SIZE // w, PREVIEW_CANVAS_SIZE // h))
        disp_w, disp_h = w * cell_zoom, h * cell_zoom
        crop = self.pil_img.crop((x0, y0, x1, y1)).resize((disp_w, disp_h), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(crop)

        off_x = (PREVIEW_CANVAS_SIZE - disp_w) // 2
        off_y = (PREVIEW_CANVAS_SIZE - disp_h) // 2
        self.preview_canvas.create_image(off_x, off_y, anchor=tk.NW, image=self._preview_photo)

        show_values = w * h <= PREVIEW_VALUE_TEXT_MAX_CELLS and cell_zoom >= 8
        if cell_zoom >= 4:
            for i in range(w + 1):
                gx = off_x + i * cell_zoom
                self.preview_canvas.create_line(gx, off_y, gx, off_y + disp_h, fill="#555555")
            for j in range(h + 1):
                gy = off_y + j * cell_zoom
                self.preview_canvas.create_line(off_x, gy, off_x + disp_w, gy, fill="#555555")

        patch = self.img_array[y0:y1, x0:x1]
        if show_values:
            for j in range(h):
                for i in range(w):
                    val = int(patch[j, i])
                    cx = off_x + i * cell_zoom + cell_zoom / 2
                    cy = off_y + j * cell_zoom + cell_zoom / 2
                    color = "white" if val < 128 else "black"
                    self.preview_canvas.create_text(cx, cy, text=str(val), fill=color, font=("", 8))

        if highlight is not None:
            hix, hiy = highlight
            if x0 <= hix < x1 and y0 <= hiy < y1:
                hx = off_x + (hix - x0) * cell_zoom
                hy = off_y + (hiy - y0) * cell_zoom
                self.preview_canvas.create_rectangle(
                    hx, hy, hx + cell_zoom, hy + cell_zoom, outline="#ffcc00", width=2,
                )

    # ------------------------------------------------------------- Panel

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        by_label: dict[str, list[Selection]] = {}
        for sel in self.selections:
            by_label.setdefault(sel.label, []).append(sel)

        for label, sels in by_label.items():
            all_values = np.concatenate([s.values for s in sels])
            parent = self.tree.insert(
                "", tk.END, text=label, values=(
                    all_values.size, f"{all_values.mean():.2f}", f"{all_values.std():.2f}",
                    int(all_values.min()), int(all_values.max()),
                ), tags=("group",),
            )
            for sel in sels:
                self.tree.insert(
                    parent, tk.END,
                    text=f"({sel.x0},{sel.y0})-({sel.x1},{sel.y1})",
                    values=(
                        sel.values.size, f"{sel.values.mean():.2f}", f"{sel.values.std():.2f}",
                        int(sel.values.min()), int(sel.values.max()),
                    ),
                    tags=("item", str(id(sel))),
                )
        self.tree.tag_configure("group", font=("", 9, "bold"))

    def _delete_selected(self) -> None:
        sel_id = self.tree.selection()
        if not sel_id:
            return
        item = self.tree.item(sel_id[0])
        tags = item["tags"]
        if "item" in tags:
            target_id = int(tags[1])
            self.selections = [s for s in self.selections if id(s) != target_id]
        else:
            label = item["text"]
            self.selections = [s for s in self.selections if s.label != label]
        self._render()
        self._refresh_tree()
        self._redraw_histogram()

    def _clear_selections(self) -> None:
        self.selections.clear()
        self.label_colors.clear()
        self._render()
        self._refresh_tree()
        self._redraw_histogram()

    def _redraw_histogram(self) -> None:
        self.ax.clear()
        by_label: dict[str, list[np.ndarray]] = {}
        for sel in self.selections:
            by_label.setdefault(sel.label, []).append(sel.values)

        for label, arrs in by_label.items():
            values = np.concatenate(arrs)
            self.ax.hist(
                values, bins=range(0, 257, 4), alpha=0.55,
                label=f"{label} (n={values.size})", color=self.label_colors[label],
            )
        self.ax.set_xlabel("Intensidad de píxel (0-255)")
        self.ax.set_ylabel("Frecuencia")
        if by_label:
            self.ax.legend(fontsize=8)
        self.hist_canvas.draw()

    def _save_histogram(self) -> None:
        if not self.selections:
            messagebox.showinfo("Sin selecciones", "No hay selecciones para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.hist_canvas.figure.savefig(path)
            self.status.set(f"Histograma guardado: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspector interactivo de píxeles con zoom, pan y selección para histograma.")
    parser.add_argument("image", nargs="?", help="Ruta de la imagen a inspeccionar (si se omite, se abre un diálogo)")
    args = parser.parse_args()

    root = tk.Tk()
    image_path = args.image
    if not image_path:
        image_path = filedialog.askopenfilename(
            title="Elegí una imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")],
        )
    if not image_path:
        return

    PixelInspectorApp(root, image_path)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()
