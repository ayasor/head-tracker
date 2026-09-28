# Anti-missile — head tracker de baja latencia

Proyecto de Computer Vision (UPF). Detecta personas con la webcam y dibuja en tiempo real una
mira sobre su cabeza, con el objetivo de **minimizar la latencia de extremo a extremo**
(captura → pantalla).

## Cómo funciona

```
[captura]    cámara V4L2 ──► último frame + timestamp del driver
[inferencia] pre-proceso en GPU + YOLO-pose (FP16, CUDA Graph) ──► NMS / end-to-end
             ──► tracker IoU ──► filtro One Euro + predicción ──► mira + HUD
[pantalla]   imshow / waitKey / vídeo (hilo principal, no frena la inferencia)
```

- **Captura en un hilo propio** que solo conserva el frame más reciente: nunca se procesan
  imágenes viejas del buffer del driver.
- **Inferencia con CUDA Graph**: una red *nano* en GPU está limitada por Python lanzando
  cientos de kernels, no por el cómputo. Grabando pre-proceso + red en un grafo la inferencia
  baja de ~14 ms a ~3 ms.
- **Tracker IoU ligero** en lugar de ByteTrack (que con ultralytics 8.4 añade ~11 ms de flujo
  óptico por frame).
- **Centro de la cabeza** a partir de nariz/ojos/orejas; si no se ven, se estima desde los
  hombros o la caja.
- **Filtro One Euro** por persona (sin temblor en reposo, poco retraso en movimiento) y
  **predicción** por velocidad constante que adelanta la mira lo que tarda en llegar a pantalla.

## Rendimiento

Medido con una RTX 5060 Laptop y webcam 640x480 @ 30 fps:

| | Original | Actual |
|---|---|---|
| FPS | 15 | ~27–29 (límite: la cámara) |
| Inferencia | 24 ms | ~3 ms |
| Captura → pantalla (mismo reloj) | ~26 ms | ~10 ms |

El HUD mide desde el timestamp del driver V4L2, así que incluye también la transferencia USB
(~30 ms), que antes no se contaba.

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.10+ y, para ir rápido, una GPU NVIDIA con CUDA. Los pesos
(`yolo26n-pose.pt`) se descargan solos la primera vez.

## Uso

```bash
python head_tracker.py                              # webcam 0, yolo26n-pose
python head_tracker.py --exposure 150 --brighten 3  # exposición manual (evita bajar a 15 fps)
python head_tracker.py --model yolo11n-pose.pt      # otro modelo
python head_tracker.py --source video.mp4 --save out.mp4
python head_tracker.py --no-show --max-frames 300   # benchmark sin ventana
```

Teclas: `q` / `ESC` salir, `p` activar/desactivar la predicción.

Opciones útiles:

| Opción | Descripción |
|---|---|
| `--exposure N` | Exposición manual (V4L2, unidades de 100 µs). La auto-exposición con poca luz baja la cámara a 15 fps. |
| `--brighten X` | Ganancia digital si la imagen queda oscura. |
| `--imgsz N` | Lado largo de la entrada de la red. |
| `--no-predict` | Desactiva la compensación de latencia. |
| `--extra-lead-ms N` | Latencia extra (exposición, monitor) a compensar. |
| `--min-cutoff`, `--beta`, `--d-cutoff` | Parámetros del filtro One Euro. |
| `--no-graph` | Desactiva CUDA Graph. |

`python head_tracker.py --help` muestra todas.

## Citar

Ver [`CITATION.cff`](CITATION.cff) (GitHub muestra el botón *Cite this repository*).

## Licencia

[MIT](LICENSE). Nota: usa [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
que se distribuye bajo AGPL-3.0.
