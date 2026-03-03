from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from predict import find_latest_weights, run_prediction


PREVIEW_SIZE = (520, 520)
IMAGE_TYPES = [
    ("Imagenes", "*.jpg *.jpeg *.png *.bmp"),
    ("Todos los archivos", "*.*"),
]


class YoloApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.repo_root = Path(__file__).resolve().parents[1]
        self.selected_image: Path | None = None
        self.preview_image: tk.PhotoImage | None = None

        self.root.title("DeepSpace YOLO")
        self.root.geometry("940x720")
        self.root.minsize(860, 640)

        default_weights = find_latest_weights(self.repo_root / "runs")
        self.weights_var = tk.StringVar(
            value=str(default_weights) if default_weights else ""
        )
        self.image_var = tk.StringVar(value="Ninguna imagen seleccionada")
        self.status_var = tk.StringVar(
            value="Selecciona una imagen y ejecuta la deteccion."
        )
        self.conf_var = tk.DoubleVar(value=0.25)
        self.save_var = tk.BooleanVar(value=True)

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(18, 18, 18, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="DeepSpace YOLO",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Selecciona una imagen, ajusta la confianza y ejecuta la prediccion.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.Frame(self.root, padding=(18, 0, 18, 18))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(body, text="Controles", padding=14)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        controls.columnconfigure(0, weight=1)

        ttk.Button(
            controls,
            text="Seleccionar imagen",
            command=self.select_image,
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            controls,
            textvariable=self.image_var,
            wraplength=250,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 12))

        ttk.Button(
            controls,
            text="Usar ultimo modelo",
            command=self.use_latest_weights,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(controls, text="Pesos del modelo").grid(
            row=3, column=0, sticky="w", pady=(14, 4)
        )
        ttk.Entry(
            controls,
            textvariable=self.weights_var,
            width=38,
        ).grid(row=4, column=0, sticky="ew")

        ttk.Button(
            controls,
            text="Buscar pesos",
            command=self.select_weights,
        ).grid(row=5, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(controls, text="Confianza minima").grid(
            row=6, column=0, sticky="w", pady=(16, 4)
        )
        ttk.Scale(
            controls,
            from_=0.05,
            to=0.95,
            variable=self.conf_var,
            orient="horizontal",
        ).grid(row=7, column=0, sticky="ew")
        self.conf_label = ttk.Label(controls, text="")
        self.conf_label.grid(row=8, column=0, sticky="w", pady=(4, 0))
        self.conf_var.trace_add("write", self._update_conf_label)
        self._update_conf_label()

        ttk.Checkbutton(
            controls,
            text="Guardar imagen anotada",
            variable=self.save_var,
        ).grid(row=9, column=0, sticky="w", pady=(14, 0))

        ttk.Button(
            controls,
            text="Ejecutar deteccion",
            command=self.run_detection,
        ).grid(row=10, column=0, sticky="ew", pady=(18, 0))

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=250,
            justify="left",
        ).grid(row=11, column=0, sticky="ew", pady=(14, 0))

        preview = ttk.LabelFrame(body, text="Vista previa", padding=14)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            preview,
            text="La imagen seleccionada o el resultado apareceran aqui.",
            anchor="center",
            justify="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _update_conf_label(self, *_args: object) -> None:
        self.conf_label.configure(text=f"Valor actual: {self.conf_var.get():.2f}")

    def select_image(self) -> None:
        initial_dir = self.repo_root / "inputs"
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            initialdir=initial_dir if initial_dir.exists() else self.repo_root,
            filetypes=IMAGE_TYPES,
        )
        if not file_path:
            return

        self.selected_image = Path(file_path)
        self.image_var.set(str(self.selected_image))
        self.status_var.set("Imagen lista para procesar.")
        self.show_preview(self.selected_image)

    def select_weights(self) -> None:
        initial_dir = self.repo_root / "runs"
        file_path = filedialog.askopenfilename(
            title="Selecciona un archivo de pesos",
            initialdir=initial_dir if initial_dir.exists() else self.repo_root,
            filetypes=[("Pesos PyTorch", "*.pt"), ("Todos los archivos", "*.*")],
        )
        if file_path:
            self.weights_var.set(file_path)

    def use_latest_weights(self) -> None:
        latest = find_latest_weights(self.repo_root / "runs")
        if latest is None:
            messagebox.showerror(
                "Sin pesos",
                "No se encontro ningun best.pt o last.pt dentro de runs/.",
            )
            return
        self.weights_var.set(str(latest))
        self.status_var.set("Modelo actualizado al ultimo checkpoint disponible.")

    def show_preview(self, image_path: Path) -> None:
        try:
            image = tk.PhotoImage(file=str(image_path))
        except tk.TclError:
            try:
                from PIL import Image, ImageTk
            except ImportError:
                self.preview_image = None
                self.preview_label.configure(
                    image="",
                    text=(
                        "No se pudo mostrar la vista previa.\n"
                        "Instala Pillow si quieres ver JPG o PNG dentro de la GUI."
                    ),
                )
                return

            pil_image = Image.open(image_path)
            pil_image.thumbnail(PREVIEW_SIZE)
            self.preview_image = ImageTk.PhotoImage(pil_image)
        else:
            self.preview_image = image.subsample(
                max(1, image.width() // PREVIEW_SIZE[0] + 1),
                max(1, image.height() // PREVIEW_SIZE[1] + 1),
            )

        self.preview_label.configure(image=self.preview_image, text="")

    def run_detection(self) -> None:
        if self.selected_image is None:
            messagebox.showwarning(
                "Falta imagen",
                "Selecciona una imagen antes de ejecutar la deteccion.",
            )
            return

        weights_text = self.weights_var.get().strip()
        weights = Path(weights_text) if weights_text else None
        self.status_var.set("Ejecutando deteccion...")
        self.root.update_idletasks()

        try:
            prediction = run_prediction(
                source=self.selected_image,
                weights=weights,
                conf=float(self.conf_var.get()),
                save=self.save_var.get(),
                run_name="gui_predict",
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status_var.set("La deteccion fallo.")
            return

        saved_files = prediction["saved_files"]
        if saved_files:
            self.show_preview(saved_files[0])

        message = (
            f"Objetos detectados: {prediction['total_boxes']}. "
            f"Modelo: {Path(prediction['weights']).name}"
        )
        if self.save_var.get():
            message += f". Resultado en: {prediction['output_dir']}"
        self.status_var.set(message)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    root = tk.Tk()
    app = YoloApp(root)
    app.run()


if __name__ == "__main__":
    main()
