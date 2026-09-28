"""
head_tracker.py — Detecta personas y dibuja una mira sobre la cabeza en tiempo real,
diseñado para minimizar la latencia de extremo a extremo.

Pipeline (3 hilos, cada uno se queda solo con el dato más reciente):
    [captura]    cámara V4L2 ──► último frame + timestamp del DRIVER
    [inferencia] pre-proceso en GPU + YOLO-pose (FP16, CUDA Graph) ──► NMS
                 ──► tracker IoU ──► filtro One Euro + predicción ──► mira + HUD
    [pantalla]   imshow / waitKey / vídeo (hilo principal, no frena la inferencia)

Uso típico:
    python head_tracker.py                         # webcam 0, yolo26n-pose
    python head_tracker.py --exposure 150          # exposición manual: evita que la cámara baje a 15 fps
    python head_tracker.py --exposure 150 --brighten 3   # ... y aclara la imagen si queda oscura
    python head_tracker.py --model yolo11n-pose.pt # YOLO11
    python head_tracker.py --source video.mp4      # probar con un vídeo

Teclas: q / ESC para salir, p para activar/desactivar la predicción.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import cv2
import numpy as np

# Índices de keypoints COCO (17 puntos) que devuelve YOLO-pose
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
HEAD_KPTS = np.array([NOSE, L_EYE, R_EYE, L_EAR, R_EAR])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Captura: un hilo que SOLO guarda el último frame
# ─────────────────────────────────────────────────────────────────────────────
class LatestFrameGrabber:
    """Lee la cámara en un hilo aparte y se queda solo con el frame más nuevo.

    Por qué: si lees con cap.read() en el mismo bucle que la inferencia, y la
    inferencia tarda más que el periodo de la cámara, los frames se acumulan en el
    buffer del driver y acabas procesando imágenes de hace 100-300 ms. Aquí los
    frames viejos se descartan: siempre procesas "el presente".

    En Linux/V4L2 el timestamp es el que pone el driver al recibir el frame por USB
    (CLOCK_MONOTONIC, el mismo reloj que time.perf_counter). Llega ~1 periodo de
    cámara antes que el retorno de cap.read(), así que la latencia medida y la
    predicción son más realistas.
    """

    def __init__(self, source, width: int, height: int, fps: int, exposure: float | None):
        self.is_file = not isinstance(source, int)
        backend = cv2.CAP_V4L2 if (not self.is_file and sys.platform.startswith("linux")) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, backend)
        if not self.is_file:
            # MJPG permite resoluciones/fps altos por USB 2.0 sin que la cámara baje los fps
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if exposure is not None:
                # Con auto-exposición y poca luz muchas webcams alargan la exposición y
                # bajan a 15 fps: +33 ms de periodo y más desenfoque. Manual lo evita.
                # En V4L2: 1 = manual; unidades del driver (normalmente 100 µs).
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de vídeo: {source}")
        self.driver_ts = backend == cv2.CAP_V4L2

        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.req_fps = src_fps
        self.file_period = 1.0 / src_fps if self.is_file else 0.0

        self._cond = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self.fps = 0.0  # fps reales entregados por la cámara
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Arranca la captura (después de cargar el modelo: un vídeo no avanza en vano)."""
        self._thread.start()
        return self

    def _loop(self):
        t_prev = None
        while self.running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                if self.is_file:
                    break
                time.sleep(0.001)
                continue
            if self.driver_ts:
                ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if 0.0 <= t - ts < 0.5:  # solo si es del mismo reloj
                    t = ts
            if t_prev is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(t - t_prev, 1e-6) if self.fps else 1.0 / max(t - t_prev, 1e-6)
            t_prev = t
            with self._cond:
                self._frame, self._t, self._seq = frame, t, self._seq + 1
                self._cond.notify_all()
            if self.is_file:  # simula una cámara real: el vídeo avanza aunque no lo leas
                time.sleep(self.file_period)
        with self._cond:
            self.running = False
            self._cond.notify_all()

    def read(self, last_seq: int):
        """Devuelve (frame, t_captura, seq) en cuanto haya uno NUEVO (sin sondeo)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self.running)
            if self._seq != last_seq:
                return self._frame, self._t, self._seq
        return None, 0.0, last_seq

    def props(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self.cap.get(cv2.CAP_PROP_FPS))

    def release(self):
        with self._cond:
            self.running = False
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Modelo: YOLO-pose "a pelo" con CUDA Graph
# ─────────────────────────────────────────────────────────────────────────────
class PoseNet:
    """Ejecuta la red de YOLO-pose sin el predictor de ultralytics.

    Una red "nano" en GPU no está limitada por el cómputo sino por Python lanzando
    ~300 kernels por frame (~14 ms). Con la forma de entrada fija se puede grabar
    todo (pre-proceso + red) en un CUDA Graph y lanzarlo de una vez (~1-3 ms).
    El pre-proceso (BGR→RGB, normalizar, redimensionar, letterbox) también va en GPU.
    """

    def __init__(self, weights: str, frame_hw: tuple[int, int], imgsz: int,
                 conf: float, iou: float, max_det: int, use_graph: bool = True):
        import torch
        import torch.nn.functional as F
        from ultralytics import YOLO

        self.torch, self.F = torch, F
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.dtype = torch.float16 if self.cuda else torch.float32
        if self.cuda:
            torch.backends.cudnn.benchmark = True  # la forma no cambia: autotuning una vez

        yolo = YOLO(weights, task="pose")
        # YOLO26 trae una cabeza "one-to-one" sin NMS: se elige ANTES de fusionar
        self.e2e = getattr(yolo.model.model[-1], "one2one_cv2", None) is not None
        if self.e2e:
            yolo.model.end2end = True
        net = yolo.model.fuse(verbose=False).eval().to(self.device, self.dtype)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.nc = len(net.names)
        self.kpt_shape = tuple(net.model[-1].kpt_shape)  # (17, 3)
        stride = int(max(net.stride))

        H, W = frame_hw
        self.frame_hw = frame_hw
        self.r = r = min(imgsz / H, imgsz / W)
        self.nh, self.nw = round(H * r), round(W * r)
        ih, iw = math.ceil(self.nh / stride) * stride, math.ceil(self.nw / stride) * stride
        self.conf, self.iou, self.max_det = conf, iou, max_det

        # Buffers estáticos (el CUDA Graph siempre lee/escribe las mismas direcciones)
        self.src = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        self.inp = torch.full((1, 3, ih, iw), 114 / 255, dtype=self.dtype, device=self.device)
        if self.cuda:
            self.host = torch.zeros((H, W, 3), dtype=torch.uint8).pin_memory()
            self.host_np = self.host.numpy()

        self.graph = None
        with torch.inference_mode():
            if self.cuda and use_graph:
                try:
                    self._capture_graph()
                except Exception as e:  # si alguna capa no es capturable, modo normal
                    print(f"[aviso] CUDA Graph no disponible ({e}); se usa ejecución normal")
                    self.graph = None
            if self.graph is None:
                for _ in range(3):
                    self.out = self._forward()
        for _ in range(2):  # calienta también el post-proceso (primera llamada lenta)
            self(np.zeros((H, W, 3), np.uint8))
        print(f"[info] modelo={weights}  device={self.device}  dtype={str(self.dtype)[6:]}  "
              f"entrada={iw}x{ih}  cuda_graph={self.graph is not None}")

    def _forward(self):
        x = self.src.permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype).mul_(1 / 255)  # BGR→RGB, 0-1
        if (self.nh, self.nw) != self.frame_hw:
            x = self.F.interpolate(x, size=(self.nh, self.nw), mode="bilinear", align_corners=False)
        self.inp[:, :, :self.nh, :self.nw] = x
        y = self.net(self.inp)
        # (1, 4+nc+17*3, anchors) con NMS pendiente, o (1, max_det, 6+17*3) si es end-to-end
        return y[0] if isinstance(y, (list, tuple)) else y

    def _capture_graph(self):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm-up: anchors, cuDNN autotuning, memoria
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self._forward()
        self.graph = g
        g.replay()
        torch.cuda.synchronize()

    def __call__(self, frame: np.ndarray):
        """Devuelve (boxes xyxy (n,4), scores (n,), kxy (n,17,2), kconf (n,17)) en px del frame."""
        torch = self.torch
        with torch.inference_mode():
            if self.cuda:
                np.copyto(self.host_np, frame)
                self.src.copy_(self.host, non_blocking=True)
            else:
                self.src.copy_(torch.from_numpy(frame))
            if self.graph is not None:
                self.graph.replay()
            else:
                self.out = self._forward()

            if self.e2e:  # filas: x1 y1 x2 y2 score clase kpts… (ya sin duplicados)
                p = self.out[0]
                keep = p[:, 4] > self.conf
                p = p[keep][:self.max_det].float()
                res = torch.cat((p[:, :5], p[:, 6:]), 1)
            else:  # filas: cx cy w h scores[nc] kpts… → NMS
                p = self.out[0].transpose(0, 1)
                scores = p[:, 4:4 + self.nc].amax(1)
                keep = scores > self.conf
                p, scores = p[keep].float(), scores[keep].float()
                xy, wh = p[:, :2], p[:, 2:4] / 2
                boxes = torch.cat((xy - wh, xy + wh), 1)
                if len(boxes):
                    import torchvision
                    idx = torchvision.ops.nms(boxes, scores, self.iou)[:self.max_det]
                    boxes, scores, p = boxes[idx], scores[idx], p[idx]
                res = torch.cat((boxes, scores[:, None], p[:, 4 + self.nc:]), 1)
            # Una sola copia GPU→CPU (una sola sincronización) por frame
            res = res.cpu().numpy()

        nk, kd = self.kpt_shape
        kpts = res[:, 5:].reshape(-1, nk, kd)
        boxes = res[:, :4] / self.r
        kxy = kpts[..., :2] / self.r
        kconf = kpts[..., 2] if kd == 3 else None
        return boxes, res[:, 4], kxy, kconf


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tracker mínimo por IoU (IDs estables para el filtro de cada persona)
# ─────────────────────────────────────────────────────────────────────────────
class IoUTracker:
    """Asociación voraz por IoU entre frames consecutivos.

    El ByteTrack de ultralytics hace además compensación de movimiento de cámara con
    flujo óptico (~10 ms/frame en CPU); con cámara fija no aporta nada y solo
    necesitamos IDs para el suavizado, así que esto basta (<0,1 ms).
    """

    def __init__(self, iou_thr: float = 0.3, max_age: float = 0.5):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: dict[int, tuple[np.ndarray, float]] = {}
        self._next = 1

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        inter = np.clip(rb - lt, 0, None).prod(2)
        area_a = (a[:, 2:] - a[:, :2]).prod(1)
        area_b = (b[:, 2:] - b[:, :2]).prod(1)
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

    def update(self, boxes: np.ndarray, t: float) -> list[int]:
        ids = [0] * len(boxes)
        tids = list(self.tracks)
        if tids and len(boxes):
            iou = self._iou(boxes, np.stack([self.tracks[k][0] for k in tids]))
            used_d, used_t = set(), set()
            for flat in np.argsort(-iou, axis=None):
                d, j = divmod(int(flat), len(tids))
                if iou[d, j] < self.iou_thr:
                    break
                if d in used_d or j in used_t:
                    continue
                ids[d] = tids[j]
                used_d.add(d)
                used_t.add(j)
        for d, box in enumerate(boxes):
            if not ids[d]:
                ids[d] = self._next
                self._next += 1
            self.tracks[ids[d]] = (box, t)
        for k in [k for k, (_, tl) in self.tracks.items() if t - tl > self.max_age]:
            del self.tracks[k]
        return ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. Suavizado: filtro One Euro (Casiez et al., CHI 2012)
# ─────────────────────────────────────────────────────────────────────────────
class OneEuroFilter:
    """Paso bajo de primer orden cuya frecuencia de corte crece con la velocidad.

    - Objeto quieto  → corte bajo (min_cutoff) → mucho suavizado, sin temblor.
    - Objeto rápido  → corte alto (min_cutoff + beta·|v|) → casi sin retraso.
    Un EMA con alfa fijo te obliga a elegir entre temblor y lag; este no.

    Además expone la velocidad filtrada (self.dx), que se usa para predecir. Su corte
    (d_cutoff) no debe ser muy bajo: con 1 Hz la velocidad llega ~160 ms tarde y la
    predicción se queda corta justo cuando más falta hace.
    """

    def __init__(self, min_cutoff=1.5, beta=0.05, d_cutoff=3.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None          # posición filtrada (px)
        self.dx = np.zeros(2)  # velocidad filtrada (px/s)
        self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # Discretización del RC: tau = 1/(2·pi·fc), alpha = dt/(dt+tau)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.t = x.copy(), t
            return self.x
        dt = max(t - self.t, 1e-4)
        self.t = t
        raw_dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * raw_dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x

    def predict(self, lead_s: float, max_shift: float) -> np.ndarray:
        """Extrapolación de velocidad constante: x(t + lead) ≈ x(t) + v·lead (acotada)."""
        shift = self.dx * lead_s
        n = float(np.linalg.norm(shift))
        if n > max_shift:
            shift *= max_shift / n
        return self.x + shift


# ─────────────────────────────────────────────────────────────────────────────
# 5. Geometría: de keypoints a "centro de la cabeza"
# ─────────────────────────────────────────────────────────────────────────────
def head_from_pose(kxy: np.ndarray, kconf: np.ndarray | None, box: np.ndarray, thr: float):
    """Estima (centro, radio) de la cabeza de una persona.

    Prioridad:
      1) Media de nariz/ojos/orejas visibles ponderada por confianza.
      2) Si la cara no se ve (de espaldas, tapada): extrapolar desde los hombros.
      3) Último recurso: parte superior de la bounding box.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    conf = kconf if kconf is not None else np.ones(len(kxy))

    # Radio aproximado: ~0.35 del ancho de hombros, o fracción de la caja
    sh_ok = conf[L_SH] > thr and conf[R_SH] > thr
    sh_w = float(np.linalg.norm(kxy[L_SH] - kxy[R_SH])) if sh_ok else 0.0
    radius = 0.35 * sh_w if sh_w > 5 else 0.18 * bw
    radius = float(np.clip(radius, 8, 0.5 * max(bw, 1)))

    vis = HEAD_KPTS[conf[HEAD_KPTS] > thr]
    if len(vis) > 0:
        w = conf[vis][:, None]
        center = (kxy[vis] * w).sum(0) / w.sum()
        return center, radius, "pose"

    if sh_ok:
        mid = (kxy[L_SH] + kxy[R_SH]) / 2
        # la cabeza queda ~0.75 anchos de hombro por encima del punto medio
        return mid - np.array([0.0, 0.75 * sh_w]), radius, "hombros"

    return np.array([(x1 + x2) / 2, y1 + 0.10 * bh]), radius, "caja"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dibujo
# ─────────────────────────────────────────────────────────────────────────────
def draw_crosshair(img, c, r, color=(0, 255, 0), thick=2):
    cx, cy = int(round(c[0])), int(round(c[1]))
    r = int(r)
    gap = max(3, r // 3)
    arm = r + max(6, r // 2)
    cv2.circle(img, (cx, cy), r, color, thick, cv2.LINE_AA)
    cv2.line(img, (cx - arm, cy), (cx - gap, cy), color, thick, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + arm, cy), color, thick, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy - gap), color, thick, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + arm), color, thick, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)


def draw_hud(img, lines):
    # Fondo negro en vez de contorno grueso: en OpenCV 5 el texto con thickness=3 cambia
    # de espaciado y el "contorno" sale desplazado (texto doble).
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines)
    cv2.rectangle(img, (4, 4), (16 + w, 10 + 22 * len(lines)), (0, 0, 0), -1)
    y = 22
    for text in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += 22


def ema(old, new, a=0.1):
    return new if old is None else (1 - a) * old + a * new


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hilo de inferencia
# ─────────────────────────────────────────────────────────────────────────────
class Stats:
    """Estado compartido entre el hilo de inferencia y el de pantalla."""

    def __init__(self, predict: bool):
        self.predict = predict
        self.stop = False
        self.fps = self.inf = self.lat = None
        self.show_delay = 0.0  # s entre "frame listo" y "frame en pantalla" (EMA)
        self.cond = threading.Condition()
        self.out = None  # (frame anotado, t_captura, t_listo, n)


def inference_loop(args, grab: LatestFrameGrabber, net: PoseNet, st: Stats):
    import torch

    tracker = IoUTracker()
    filters: dict[int, OneEuroFilter] = {}
    lut = None
    if args.brighten != 1.0:
        lut = np.clip(np.arange(256) * args.brighten, 0, 255).astype(np.uint8)
    extra = args.extra_lead_ms / 1000.0
    seq, n = 0, 0
    t_prev = time.perf_counter()

    with torch.inference_mode():
        while not st.stop:
            frame, t_cap, seq = grab.read(seq)
            if frame is None:
                break
            t0 = time.perf_counter()
            if lut is not None:
                cv2.LUT(frame, lut, dst=frame)
            boxes, scores, kxy, kcf = net(frame)
            t_inf = time.perf_counter() - t0

            now = time.perf_counter()
            # Cuánto llegará tarde la mira: captura→ahora + ahora→pantalla (+ extra de
            # cámara/monitor que no se puede medir). La predicción adelanta eso.
            lead = (now - t_cap) + st.show_delay + extra if st.predict else 0.0

            ids = tracker.update(boxes, t_cap) if not args.no_track else [0] * len(boxes)
            for i in range(len(boxes)):
                center, radius, how = head_from_pose(
                    kxy[i], None if kcf is None else kcf[i], boxes[i], args.kpt_thr)
                tid = ids[i]
                if tid:
                    f = filters.setdefault(tid, OneEuroFilter(args.min_cutoff, args.beta, args.d_cutoff))
                    f(center, t_cap)  # el tiempo relevante es CUÁNDO se tomó la imagen
                    center = f.predict(lead, 2.0 * radius) if st.predict else f.x

                color = (0, 255, 0) if how == "pose" else (0, 165, 255)
                draw_crosshair(frame, center, radius, color)
                label = f"#{tid}" if tid else ""
                cv2.putText(frame, f"{label} {how}", (int(center[0] + radius + 4), int(center[1] - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            for tid in [k for k in filters if k not in tracker.tracks]:
                del filters[tid]

            st.fps = ema(st.fps, 1.0 / max(now - t_prev, 1e-6))
            st.inf = ema(st.inf, t_inf * 1000)
            t_prev = now
            draw_hud(frame, [
                f"FPS {st.fps:5.1f}  (camara {grab.fps:4.1f})",
                f"inferencia {st.inf:5.1f} ms",
                f"captura->pantalla {st.lat or 0:5.1f} ms",
                f"prediccion {'ON' if st.predict else 'OFF'} (p)",
            ])

            n += 1
            if n == 60 and not grab.is_file and grab.fps < 0.8 * grab.req_fps:
                print(f"[aviso] la cámara solo entrega {grab.fps:.1f} fps de {grab.req_fps:.0f}: probablemente "
                      f"la auto-exposición alarga la exposición. Prueba --exposure 150 (y --brighten 2-3).")
            with st.cond:
                st.out = (frame, t_cap, time.perf_counter(), n)
                st.cond.notify_all()
            if args.max_frames and n >= args.max_frames:
                break

    with st.cond:
        st.stop = True
        st.cond.notify_all()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Principal: pantalla / vídeo en el hilo principal (HighGUI lo prefiere así)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="índice de cámara (0, 1…) o ruta de vídeo")
    ap.add_argument("--model", default="yolo26n-pose.pt", help="yolo26n/s/m-pose.pt, yolo11n-pose.pt…")
    ap.add_argument("--imgsz", type=int, default=640, help="lado largo de entrada de la red")
    ap.add_argument("--width", type=int, default=640, help="ancho pedido a la cámara")
    ap.add_argument("--height", type=int, default=480, help="alto pedido a la cámara")
    ap.add_argument("--fps", type=int, default=60, help="fps pedidos a la cámara")
    ap.add_argument("--exposure", type=float, default=None,
                    help="exposición manual (V4L2: unidades de 100 µs, p.ej. 150 = 15 ms)")
    ap.add_argument("--brighten", type=float, default=1.0, help="ganancia digital si la imagen queda oscura")
    ap.add_argument("--conf", type=float, default=0.4, help="confianza mínima de detección")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU del NMS")
    ap.add_argument("--max-det", type=int, default=10)
    ap.add_argument("--kpt-thr", type=float, default=0.5, help="confianza mínima de keypoint")
    ap.add_argument("--no-graph", action="store_true", help="desactivar CUDA Graph")
    ap.add_argument("--no-track", action="store_true", help="sin tracker (sin IDs ni suavizado)")
    ap.add_argument("--no-predict", action="store_true", help="no compensar la latencia extrapolando")
    ap.add_argument("--extra-lead-ms", type=float, default=0.0,
                    help="latencia no medible a compensar además (exposición, monitor…)")
    ap.add_argument("--min-cutoff", type=float, default=1.5, help="One Euro: suavizado en reposo (Hz)")
    ap.add_argument("--beta", type=float, default=0.05, help="One Euro: reactividad a la velocidad")
    ap.add_argument("--d-cutoff", type=float, default=3.0, help="One Euro: suavizado de la velocidad (Hz)")
    ap.add_argument("--no-show", action="store_true", help="no abrir ventana (benchmark)")
    ap.add_argument("--save", default="", help="guardar el resultado en este .mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    grab = LatestFrameGrabber(source, args.width, args.height, args.fps, args.exposure)
    W, H, cam_fps = grab.props()
    print(f"[info] cámara {W}x{H} @ {cam_fps:.0f} fps (pedido)")
    net = PoseNet(args.model, (H, W), args.imgsz, args.conf, args.iou, args.max_det, not args.no_graph)
    grab.start()

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), cam_fps or 30, (W, H))

    st = Stats(predict=not args.no_predict)
    worker = threading.Thread(target=inference_loop, args=(args, grab, net, st), daemon=True)
    worker.start()

    last_n = 0
    try:
        while True:
            with st.cond:
                st.cond.wait_for(lambda: st.stop or (st.out is not None and st.out[3] != last_n))
                if st.out is None or st.out[3] == last_n:
                    break  # parado y sin frame nuevo
                frame, t_cap, t_ready, last_n = st.out
            if not args.no_show:
                cv2.imshow("head tracker", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("p"):
                    st.predict = not st.predict
            t_shown = time.perf_counter()
            st.show_delay = 0.9 * st.show_delay + 0.1 * (t_shown - t_ready)
            st.lat = ema(st.lat, (t_shown - t_cap) * 1000)
            if writer is not None:  # después de mostrar: no retrasa la pantalla
                writer.write(frame)
    except KeyboardInterrupt:
        pass
    finally:
        st.stop = True
        worker.join(timeout=2.0)
        grab.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        if st.inf is not None:
            print(f"[resumen] {last_n} frames | FPS≈{st.fps:.1f} | inferencia≈{st.inf:.1f} ms "
                  f"| captura→pantalla≈{st.lat or 0:.1f} ms")


if __name__ == "__main__":
    main()
