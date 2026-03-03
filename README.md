# DeepSpace YOLO

Proyecto base para entrenar y usar localmente un detector de objetos con YOLO a partir del dataset `DeepSpaceYoloDataset.zip`.

## Estructura recomendada

```text
yoloV1/
|-- DeepSpaceYoloDataset.zip
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- inputs/
|   `-- .gitkeep
|-- configs/
|   `-- deepspace_dataset.yaml
|-- data/
|   |-- raw/
|   `-- yolo/
|       |-- images/
|       |   |-- train/
|       |   |-- val/
|       |   `-- test/
|       `-- labels/
|           |-- train/
|           |-- val/
|           `-- test/
`-- src/
    |-- prepare_dataset.py
    |-- train.py
    `-- predict.py
```

## Flujo de trabajo

1. Instala dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepara el dataset a partir del `.zip`:

```powershell
python src/prepare_dataset.py --class-name space_object
```

Ese script:
- descomprime el dataset dentro de `data/raw/`
- verifica que cada imagen tenga su etiqueta
- divide el dataset en `train`, `val` y `test`
- genera `configs/deepspace_dataset.yaml`

Si ya tienes tus propias imagenes etiquetadas en una carpeta con estructura `images/` y `labels/`, puedes usarla directamente:

```powershell
python src/prepare_dataset.py --dataset-dir ruta\a\tu_dataset --class-names star planet galaxy nebula
```

Si el dataset tiene varias clases y no pasas `--class-names`, el script generara nombres temporales como `class_0`, `class_1`, etc.

3. Entrena el modelo:

```powershell
python src/train.py --epochs 50 --imgsz 608 --batch 16
```

Para una prueba rapida antes del entrenamiento largo:

```powershell
python src/train.py --epochs 1 --fraction 0.1 --imgsz 608 --batch 16 --run-name smoke_test
```

En Windows, el script usa `workers=0` por defecto para evitar bloqueos del dataloader. Si todo va bien y quieres exprimir un poco mas el CPU, puedes probar `--workers 2` o `--workers 4`.

4. Prueba una imagen local:

Coloca la imagen en `inputs/`, por ejemplo `inputs/mi_prueba.jpg`, y ejecuta:

```powershell
python src/predict.py --weights runs/deepspace_yolov8n_rtx4050_w0_ep10/weights/best.pt --source inputs/mi_prueba.jpg --save
```

La imagen anotada se guardara en `outputs/predict/`.

5. Abre una interfaz grafica simple:

```powershell
python src/gui.py
```

La interfaz permite:
- seleccionar cualquier imagen con un explorador de archivos
- usar automaticamente el ultimo modelo encontrado en `runs/`
- ajustar la confianza minima
- guardar y previsualizar la imagen anotada

## Punto importante del dataset actual

Revise el `.zip` y actualmente todas las etiquetas usan solo la clase `0`. Eso significa que el modelo, por ahora, solo puede aprender una categoria general.

Si quieres detectar tipos distintos como `planet`, `asteroid`, `moon`, etc., entonces tus archivos `.txt` deben usar varios IDs de clase y el YAML debe incluir todos esos nombres.

## Archivo de dataset esperado

Despues de correr `prepare_dataset.py`, el archivo `configs/deepspace_dataset.yaml` quedara con esta forma:

```yaml
path: data/yolo
train: images/train
val: images/val
test: images/test
names:
  0: space_object
```
