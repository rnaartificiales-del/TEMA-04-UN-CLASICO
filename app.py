import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

def find_path(*parts):
    for root in (BASE_DIR, os.path.dirname(BASE_DIR)):
        candidate = os.path.join(root, *parts)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, BASE_DIR)
    return os.path.join(*parts)

# Para usar el video de prueba:  VIDEO_SOURCE=input/input.mp4 python app.py
# Comando para verlo $env:VIDEO_SOURCE="input/input.mp4"; py -3.12 app.py
_source = os.environ.get("VIDEO_SOURCE", "0")
if _source.isdigit():
    SOURCE = int(_source)
elif os.path.isabs(_source):
    SOURCE = _source
else:
    SOURCE = find_path(*_source.replace("\\", "/").split("/"))

OBJ_THRESH = 0.6
P_THRESH = 0.6
NMS_THRESH = 0.5
DETECT_EVERY = 60          # cada cuántos fotogramas se vuelve a detectar con YOLO
STREAM_MAX_WIDTH = 960     # ancho máximo del video que se envía al navegador
JPEG_QUALITY = 80

np.random.seed(999)

# Modelo YOLOv3 (COCO)
with open(find_path("yolov3", "coco.names"), "rt") as f:
    CLASSES = f.read().rstrip("\n").split("\n")
COLORS = np.random.randint(0, 255, (len(CLASSES), 3))

model = cv2.dnn.readNetFromDarknet(
    find_path("yolov3", "yolov3.cfg"),
    find_path("yolov3", "yolov3.weights"),
)
OUTPUT_NAMES = model.getUnconnectedOutLayersNames()


def where_is_it(frame, outputs):
    """Extrae cajas, probabilidades y clases de las salidas de YOLO."""
    frame_h, frame_w = frame.shape[:2]
    bboxes, probs, class_ids = [], [], []
    for preds in outputs:
        hits = np.any(preds[:, 5:] > P_THRESH, axis=1) & (preds[:, 4] > OBJ_THRESH)
        for i in np.where(hits)[0]:
            pred = preds[i, :]
            center_x = int(pred[0] * frame_w)
            center_y = int(pred[1] * frame_h)
            width = int(pred[2] * frame_w)
            height = int(pred[3] * frame_h)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            bboxes.append([left, top, width, height])
            probs.append(float(np.max(pred[5:])))
            class_ids.append(int(np.argmax(pred[5:])))
    return bboxes, probs, class_ids


def draw_box(frame, class_id, x, y, w, h):
    col = [int(c) for c in COLORS[class_id, :]]
    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
    cv2.putText(frame, CLASSES[class_id], (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

# Procesador de video (corre en un hilo aparte)

class VideoProcessor:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.status = "iniciando"       # iniciando | activo | error (q no slaga error plis)
        self.message = ""
        self.fps = 0.0
        self.counts = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _publish(self, frame):
        h, w = frame.shape[:2]
        if w > STREAM_MAX_WIDTH:
            scale = STREAM_MAX_WIDTH / w
            frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def _run(self):
        vid = cv2.VideoCapture(SOURCE)
        if not vid.isOpened():
            self.status = "error"
            self.message = "No se pudo abrir la fuente de video"
            return

        is_file = isinstance(SOURCE, str)
        src_fps = vid.get(cv2.CAP_PROP_FPS) or 25
        frame_delay = (1.0 / src_fps) if is_file else 0

        mtracker = None
        track_classes = []
        count = 0
        last_time = time.time()

        while True:
            t0 = time.time()
            ok, frame = vid.read()

            if not ok:
                if is_file:
                    # el video terminó: se reinicia en bucle escap jeje o control c
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    mtracker, track_classes, count = None, [], 0
                    continue
                self.status = "error"
                self.message = "Se perdió la señal de la cámara"
                break

            counts = {}

            if mtracker is None or count >= DETECT_EVERY:
                #  Detección n YOLO 
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
                model.setInput(blob)
                outputs = model.forward(OUTPUT_NAMES)
                bboxes, probs, class_ids = where_is_it(frame, outputs)

                if len(bboxes) > 0:
                    idxs = np.array(cv2.dnn.NMSBoxes(bboxes, probs, P_THRESH, NMS_THRESH)).flatten()
                    mtracker = cv2.legacy.MultiTracker_create()
                    track_classes = []
                    for i in idxs:
                        x, y, w, h = [int(v) for v in bboxes[i]]
                        mtracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, (x, y, w, h))
                        track_classes.append(class_ids[i])
                        draw_box(frame, class_ids[i], x, y, w, h)
                        name = CLASSES[class_ids[i]]
                        counts[name] = counts.get(name, 0) + 1
                    count = 1
                else:
                    mtracker, track_classes, count = None, [], 0
            else:
                #  Seguimiento entre detecciones 
                is_tracking, boxes = mtracker.update(frame)
                if is_tracking:
                    for cid, bbox in zip(track_classes, boxes):
                        x, y, w, h = [int(v) for v in bbox]
                        draw_box(frame, cid, x, y, w, h)
                        name = CLASSES[cid]
                        counts[name] = counts.get(name, 0) + 1
                    count += 1
                else:
                    mtracker, track_classes, count = None, [], 0

            #  FPS (van a ser malisimos por mi lap xd)
            now = time.time()
            inst = 1.0 / max(now - last_time, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            last_time = now

            self.counts = counts
            self.status = "activo"
            self._publish(frame)

            # para archivos, se respeta la velocidad original del video
            elapsed = time.time() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)

    def stream(self):
        last = None
        while True:
            with self.cond:
                self.cond.wait(timeout=2)
                jpg = self.jpeg
            if jpg is None or jpg is last:
                continue
            last = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

processor = VideoProcessor()
# Flask(servidor para la web)
app = Flask(__name__)

@app.route("/")
def index():
    source_label = "Webcam" if isinstance(SOURCE, int) else os.path.basename(SOURCE)
    return render_template("index.html", source_label=source_label)


@app.route("/video_feed")
def video_feed():
    return Response(processor.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(
        status=processor.status,
        message=processor.message,
        fps=round(processor.fps, 1),
        total=sum(processor.counts.values()),
        objects=processor.counts,
    )

if __name__ == "__main__":
    processor.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)