"""
Detector de objetos en imagenes, con ventana tipo app de Android.

Usa el mismo YOLOv3 y OpenCV que tracker.py. No necesita navegador ni nada extra:
solo tkinter (ya viene con Python), OpenCV y numpy.

Correr:   py -3.12 app_escritorio.py
"""
import base64
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

# ---------------------------------------------------------------
# Rutas (OpenCV en Windows no abre rutas con acentos, por eso se
# trabaja desde la carpeta del archivo con rutas relativas)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def find_path(*parts):
    """Busca en esta carpeta y, si no esta, en la de arriba."""
    for root in (BASE_DIR, os.path.dirname(BASE_DIR)):
        candidate = os.path.join(root, *parts)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, BASE_DIR)
    return os.path.join(*parts)


def imread_unicode(path):
    """cv2.imread no lee rutas con acentos; esto si."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------
W, H = 380, 690            # tamano de la "pantalla del celular"
MIN_CONF = 0.30            # lo minimo que se guarda del modelo
DEFAULT_CONF = 0.60        # umbral inicial (el mismo de tracker.py)
NMS_THRESH = 0.5
MAX_SIDE = 1600            # las fotos mas grandes se reducen antes de analizar

# Colores (los mismos de la app web y de la de Flutter)
NIGHT = "#12102A"
PANEL = "#1B1840"
LINE = "#3A3566"
MUTED = "#B3B0C8"
WHITE = "#FFFFFF"
OK = "#4ADE80"
WARN = "#FBBF24"
ERR = "#F87171"
NIGHT_BGR = (42, 16, 18)

PALETTE = [  # BGR
    (250, 165, 96), (238, 211, 34), (250, 139, 167), (153, 114, 244),
    (36, 191, 251), (128, 222, 74), (60, 146, 251),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def font(size, style="", family="Segoe UI"):
    # tamano negativo = en pixeles (no cambia con la escala de Windows)
    return (family, -size, style) if style else (family, -size)


# ---------------------------------------------------------------
# Modelo YOLOv3
# ---------------------------------------------------------------
class Model:
    def __init__(self):
        self.net = None
        self.classes = []
        self.out_names = None

    def load(self):
        if not hasattr(cv2.dnn, "readNetFromDarknet"):
            raise RuntimeError(
                "Tu OpenCV es muy nuevo. Instala: pip install opencv-contrib-python==4.10.0.84"
            )
        with open(find_path("yolov3", "coco.names"), "rt", encoding="utf-8") as f:
            self.classes = f.read().rstrip("\n").split("\n")
        self.net = cv2.dnn.readNetFromDarknet(
            find_path("yolov3", "yolov3.cfg"),
            find_path("yolov3", "yolov3.weights"),
        )
        self.out_names = self.net.getUnconnectedOutLayersNames()

    def run(self, img):
        """Devuelve todas las cajas con confianza minima; luego se filtran."""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.out_names)

        boxes, objs, probs, cids = [], [], [], []
        for preds in outputs:
            cls = preds[:, 5:]
            cid = np.argmax(cls, axis=1)
            cp = cls[np.arange(len(cls)), cid]
            keep = np.where((preds[:, 4] > MIN_CONF) & (cp > MIN_CONF))[0]
            for i in keep:
                p = preds[i]
                cx, cy, bw, bh = p[0] * w, p[1] * h, p[2] * w, p[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                objs.append(float(p[4]))
                probs.append(float(cp[i]))
                cids.append(int(cid[i]))
        return {"boxes": boxes, "objs": objs, "probs": probs, "cids": cids}


def select(raw, thr):
    """Filtra por confianza y quita cajas repetidas (NMS)."""
    idx = [
        i for i in range(len(raw["boxes"]))
        if raw["objs"][i] >= thr and raw["probs"][i] >= thr
    ]
    if not idx:
        return []
    boxes = [raw["boxes"][i] for i in idx]
    probs = [raw["probs"][i] for i in idx]
    keep = np.array(cv2.dnn.NMSBoxes(boxes, probs, thr, NMS_THRESH)).flatten()
    return [
        (boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3], raw["cids"][idx[k]], probs[k])
        for k in keep
    ]


def annotate(img, dets, classes, scale=1.0, thick=2, fscale=0.5):
    out = img.copy()
    for x, y, w, h, cid, score in dets:
        color = PALETTE[cid % len(PALETTE)]
        x1, y1 = int(x * scale), int(y * scale)
        x2, y2 = int((x + w) * scale), int((y + h) * scale)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = "%s %d%%" % (classes[cid], int(score * 100))
        (tw, th), _ = cv2.getTextSize(label, FONT, fscale, 1)
        ty = max(y1, th + 8)
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
        cv2.putText(out, label, (x1 + 4, ty - 5), FONT, fscale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def to_photo(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


def make_background(alphas):
    """Foto del Tec recortada al tamano de la pantalla y oscurecida."""
    img = imread_unicode(os.path.join(BASE_DIR, "static", "tec.jpg"))
    night = np.array(NIGHT_BGR, dtype=np.float32)
    if img is None:
        return np.full((H, W, 3), NIGHT_BGR, np.uint8)
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA)
    x0 = int((img.shape[1] - W) * 0.62)   # se recorre hacia donde esta el letrero
    img = img[:H, x0:x0 + W].astype(np.float32)
    a = np.interp(np.linspace(0, 1, H), [0, 0.45, 1], alphas)[:, None, None]
    return (img * (1 - a) + night * a).astype(np.uint8)


def rrect(c, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Detector")
        root.configure(bg=NIGHT)
        root.resizable(False, False)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d+%d+%d" % (W, H, (sw - W) // 2, max(0, (sh - H) // 2 - 20)))

        self.c = tk.Canvas(root, width=W, height=H, bg=NIGHT, highlightthickness=0, bd=0)
        self.c.pack()

        self.model = Model()
        self.q = queue.Queue()
        self.ready = False
        self.busy = False
        self.error = None
        self.toast = None
        self.screen = "home"

        self.img = None          # imagen analizada (BGR)
        self.raw = None          # detecciones sin filtrar
        self.ms = 0
        self.thr = DEFAULT_CONF
        self.dragging = False
        self._photos = []

        self.bg_home = to_photo(make_background([0.90, 0.58, 0.94]))
        self.bg_detect = to_photo(make_background([0.95, 0.93, 0.96]))

        self.c.bind("<B1-Motion>", self.on_drag)
        self.c.bind("<ButtonRelease-1>", lambda e: setattr(self, "dragging", False))

        threading.Thread(target=self._load_model, daemon=True).start()
        self.show_home()
        self.poll()

    # ---------------- hilo del modelo ----------------
    def _load_model(self):
        try:
            self.model.load()
            self.q.put(("model_ok",))
        except Exception as e:
            msg = str(e)
            if "yolov3" in msg or "NetParameter" in msg:
                msg = "No encontre la carpeta yolov3 junto a este archivo."
            self.q.put(("model_err", msg))

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "model_ok":
                    self.ready = True
                elif kind == "model_err":
                    self.error = msg[1]
                elif kind == "done":
                    _, self.img, self.raw, self.ms = msg
                    self.busy = False
                elif kind == "err":
                    self.busy = False
                    self.error = "No pude analizar la imagen: " + msg[1]
                if self.screen == "detect":
                    self.render()
        except queue.Empty:
            pass
        self.root.after(80, self.poll)

    # ---------------- utilidades de dibujo ----------------
    def bind_tag(self, tag, fn):
        self.c.tag_bind(tag, "<Button-1>", lambda e: fn())
        self.c.tag_bind(tag, "<Enter>", lambda e: self.c.config(cursor="hand2"))
        self.c.tag_bind(tag, "<Leave>", lambda e: self.c.config(cursor=""))

    def nav_pill(self):
        rrect(self.c, W / 2 - 60, 676, W / 2 + 60, 680, 2, fill="#CFCBE0", outline="")

    def pill(self, x1, y1, x2, y2, text, tag, filled):
        c = self.c
        if filled:
            rrect(c, x1, y1, x2, y2, (y2 - y1) / 2, fill=WHITE, outline="", tags=(tag,))
            fg = NIGHT
        else:
            rrect(c, x1, y1, x2, y2, (y2 - y1) / 2, fill=NIGHT, outline=LINE, tags=(tag,))
            fg = WHITE
        c.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill=fg,
                      font=font(14, "bold"), tags=(tag,))

    # ---------------- pantalla de inicio ----------------
    def show_home(self):
        self.screen = "home"
        c = self.c
        c.delete("all")
        c.create_image(0, 0, image=self.bg_home, anchor="nw")

        # logo
        c.create_oval(24, 24, 52, 52, outline=WHITE, width=2)
        c.create_oval(34, 34, 42, 42, fill=WHITE, outline="")
        c.create_text(62, 38, text="DETECTOR", fill=WHITE, anchor="w",
                      font=font(17, family="Segoe UI Light"))

        c.create_text(W / 2, 250, text="[de·tek·tor] sustantivo", fill=WHITE,
                      font=font(19, "italic", "Georgia"))
        c.create_text(W / 2, 312, text="DETECTOR", fill=WHITE,
                      font=font(52, family="Segoe UI Light"))
        c.create_text(W / 2, 382,
                      text="Detección de objetos en imágenes\nInstituto Tecnológico de Tláhuac II",
                      fill=MUTED, justify="center", font=font(14))

        self.pill(28, 570, W - 28, 622, "Comenzar   →", "go", True)
        self.bind_tag("go", self.show_detect)
        self.nav_pill()

    # ---------------- pantalla de deteccion ----------------
    def show_detect(self):
        self.screen = "detect"
        c = self.c
        c.delete("all")
        c.create_image(0, 0, image=self.bg_detect, anchor="nw")

        # barra superior
        c.create_text(28, 30, text="←", fill=WHITE, font=font(26), tags=("back",))
        self.bind_tag("back", self.show_home)
        c.create_text(60, 30, text="Detector", fill=WHITE, anchor="w",
                      font=font(20, family="Segoe UI Light"))

        # tarjeta de la imagen y tarjeta de resultados
        rrect(c, 20, 62, 360, 362, 24, fill=PANEL, outline=LINE)
        rrect(c, 20, 432, 360, 664, 24, fill=PANEL, outline=LINE)

        # botones
        self.pill(20, 374, 126, 418, "Galería", "btn_gallery", True)
        self.pill(137, 374, 243, 418, "Ejemplo", "btn_example", False)
        self.pill(254, 374, 360, 418, "Guardar", "btn_save", False)
        self.bind_tag("btn_gallery", self.pick_image)
        self.bind_tag("btn_example", self.use_example)
        self.bind_tag("btn_save", self.save_result)

        self.nav_pill()
        self.render()

    def render(self):
        """Redibuja lo que cambia (imagen, numeros, lista, control)."""
        c = self.c
        c.delete("dyn")
        self._photos = []

        # estado arriba a la derecha
        if self.error and not self.ready:
            color, text = ERR, "Error"
        elif not self.ready:
            color, text = WARN, "Cargando modelo..."
        elif self.busy:
            color, text = WARN, "Analizando..."
        else:
            color, text = OK, "Listo"
        c.create_text(W - 20, 30, text=text, fill=color, anchor="e",
                      font=font(13, "bold"), tags=("dyn",))
        tw = 8 + len(text) * 7
        c.create_oval(W - 20 - tw - 14, 25, W - 20 - tw - 6, 33, fill=color, outline="", tags=("dyn",))

        dets = []
        if self.img is not None and self.raw is not None:
            dets = select(self.raw, self.thr)

        # ---- imagen ----
        if self.img is None:
            if self.error and not self.ready:
                msg = self.error
            elif not self.ready:
                msg = "Cargando el modelo..."
            elif self.busy:
                msg = "Analizando..."
            else:
                msg = "Elige una imagen para detectar\nlos objetos que hay en ella"
            c.create_text(W / 2, 212, text="▣", fill=MUTED,
                          font=font(46), tags=("dyn",))
            c.create_text(W / 2, 290, text=msg, fill=MUTED, justify="center",
                          width=290, font=font(14), tags=("dyn",))
        else:
            h, w = self.img.shape[:2]
            s = min(336 / w, 296 / h)
            dw, dh = max(1, int(w * s)), max(1, int(h * s))
            small = cv2.resize(self.img, (dw, dh), interpolation=cv2.INTER_AREA)
            small = annotate(small, dets, self.model.classes, scale=s, thick=2, fscale=0.38)
            photo = to_photo(small)
            self._photos.append(photo)
            c.create_image(190, 212, image=photo, tags=("dyn",))
            if self.busy:
                c.create_text(190, 212, text="Analizando...", fill=WHITE,
                              font=font(18, "bold"), tags=("dyn",))

        # ---- mensajes (error o aviso) ----
        note = self.error if (self.error and self.ready) else self.toast
        if note:
            c.create_text(W / 2, 425, text=note, fill=ERR if self.error else OK,
                          font=font(12), width=340, tags=("dyn",))

        # ---- resultados ----
        c.create_text(40, 452, text="Objetos detectados", fill=MUTED, anchor="w",
                      font=font(12), tags=("dyn",))
        c.create_text(40, 480, text=str(len(dets)), fill=WHITE, anchor="w",
                      font=font(34, family="Segoe UI Light"), tags=("dyn",))
        c.create_text(200, 452, text="Tiempo de análisis", fill=MUTED, anchor="w",
                      font=font(12), tags=("dyn",))
        c.create_text(200, 480, text="%d ms" % self.ms if self.img is not None else "—",
                      fill=WHITE, anchor="w", font=font(34, family="Segoe UI Light"), tags=("dyn",))

        # control de confianza
        c.create_text(40, 526, text="Confianza mínima: %d%%" % round(self.thr * 100),
                      fill=MUTED, anchor="w", font=font(12), tags=("dyn",))
        x0, x1, y = 44, 336, 552
        kx = x0 + (self.thr - 0.3) / 0.6 * (x1 - x0)
        rrect(c, 24, 536, 356, 568, 8, fill=PANEL, outline="", tags=("dyn", "sl_hit"))
        c.create_line(x0, y, x1, y, fill=LINE, width=4, capstyle="round", tags=("dyn",))
        c.create_line(x0, y, kx, y, fill=WHITE, width=4, capstyle="round", tags=("dyn",))
        c.create_oval(kx - 9, y - 9, kx + 9, y + 9, fill=WHITE, outline="", tags=("dyn",))
        c.tag_bind("sl_hit", "<Button-1>", self.on_slider_press)

        # lista
        c.create_text(40, 586, text="En la imagen", fill=WHITE, anchor="w",
                      font=font(14, "bold"), tags=("dyn",))
        counts = {}
        for d in dets:
            name = self.model.classes[d[4]]
            counts[name] = counts.get(name, 0) + 1
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        if not items:
            c.create_text(40, 612, text="Sin detecciones", fill=MUTED, anchor="w",
                          font=font(13), tags=("dyn",))
        else:
            shown = items[:3]
            rest = len(items) - len(shown)
            rows = [(n.capitalize(), str(k)) for n, k in shown]
            if rest > 0:
                rows[-1] = ("y %d tipos más" % (rest + 1), "")
            for i, (name, k) in enumerate(rows):
                yy = 610 + i * 20
                c.create_text(40, yy, text=name, fill=WHITE, anchor="w",
                              font=font(13), tags=("dyn",))
                c.create_text(340, yy, text=k, fill=WHITE, anchor="e",
                              font=font(13, "bold"), tags=("dyn",))

    # ---------------- control de confianza ----------------
    def _set_thr_from_x(self, x):
        v = 0.3 + (min(max(x, 44), 336) - 44) / (336 - 44) * 0.6
        self.thr = round(round(v / 0.05) * 0.05, 2)
        self.thr = min(0.9, max(0.3, self.thr))
        self.render()

    def on_slider_press(self, event):
        self.dragging = True
        self._set_thr_from_x(event.x)

    def on_drag(self, event):
        if self.dragging and self.screen == "detect":
            self._set_thr_from_x(event.x)

    # ---------------- acciones ----------------
    def flash(self, text):
        self.toast = text
        self.error = self.error if not self.ready else None
        self.render()
        self.root.after(2500, self._clear_toast)

    def _clear_toast(self):
        self.toast = None
        if self.screen == "detect":
            self.render()

    def analyze(self, img):
        if self.busy or not self.ready or img is None:
            return
        h, w = img.shape[:2]
        m = max(h, w)
        if m > MAX_SIDE:
            img = cv2.resize(img, (int(w * MAX_SIDE / m), int(h * MAX_SIDE / m)),
                             interpolation=cv2.INTER_AREA)
        self.busy = True
        self.error = None
        self.render()

        def work():
            try:
                t0 = time.time()
                raw = self.model.run(img)
                self.q.put(("done", img, raw, int((time.time() - t0) * 1000)))
            except Exception as e:
                self.q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def pick_image(self):
        if not self.ready or self.busy:
            return
        path = filedialog.askopenfilename(
            title="Elige una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            self.error = "No pude leer esa imagen"
            self.render()
            return
        self.analyze(img)

    def use_example(self):
        if not self.ready or self.busy:
            return
        img = imread_unicode(os.path.join(BASE_DIR, "ejemplo.jpg"))
        if img is None:
            video = find_path("input", "input.mp4")
            if os.path.exists(video):
                cap = cv2.VideoCapture(video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
                ok, frame = cap.read()
                cap.release()
                img = frame if ok else None
        if img is None:
            self.error = "No encontré una imagen de ejemplo (falta input/input.mp4)"
            self.render()
            return
        self.analyze(img)

    def save_result(self):
        if self.img is None or self.raw is None:
            self.flash("Primero analiza una imagen")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado", defaultextension=".png", initialfile="resultado.png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")],
        )
        if not path:
            return
        h, w = self.img.shape[:2]
        m = max(h, w)
        out = annotate(self.img, select(self.raw, self.thr), self.model.classes,
                       scale=1.0, thick=max(2, m // 400), fscale=max(0.5, m / 1400))
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, out)
        if ok:
            buf.tofile(path)
            self.flash("Imagen guardada")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
