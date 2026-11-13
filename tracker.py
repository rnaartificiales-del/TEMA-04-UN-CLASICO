# Se importan las librerías, cv2, numoy y matplotlib 
import cv2, os 
import numpy as np
import matplotlib.pyplot as plt

# Se definen objectness, probabilidad y NMS para limpiar y quedarnos con los recuadros exactos
OBJ_THRESH = .6
P_THRESH = .6
NMS_THRESH = .5

# Se fija la semilla, para establecer un punto de partida 
np.random.seed(999)

# Se cargan los pesos, la configuración y las clases de YOLOv3 COCO

# Se importan los nombres de las clases
with open('yolov3/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
colors = np.random.randint(0, 255, (len(classes), 3))

# Indica los archivos de configuración, pesos del modelo y la red
cfg = 'yolov3/yolov3.cfg'
weights = 'yolov3/yolov3.weights'

# Aquí se carga el modelo
model = cv2.dnn.readNetFromDarknet(cfg, weights)

# Se extraen los nombres de las capas de salida
layersNames = model.getLayerNames()
outputNames = model.getUnconnectedOutLayersNames()

# Define la función que extrae las coordenadas de los objetos si la detección tiene éxito
def where_is_it(frame, outputs):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    bboxes, probs, class_ids = [], [], []
    for preds in outputs:
        hits = np.any(preds[:, 5:] > P_THRESH, axis=1) & (preds[:, 4] > OBJ_THRESH)
        
        # Guarda la probabilidad y las coordenadas de la caja si el objectness y la probabilidad superan sus umbrales
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
            class_ids.append(np.argmax(pred[5:]))
    return bboxes, probs, class_ids
    
# Abre la captura de video e inicia el VideoWriter
vid = cv2.VideoCapture(0)
vid_w, vid_h = int(vid.get(3)), int(vid.get(4))
out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                    vid.get(cv2.CAP_PROP_FPS), (vid_w, vid_h))

# Se comprueba si se inicio correctamente
assert vid.isOpened()

# Se inicia el contador desde cero
count = 0

# Se crea una nueva ventana donde se muestra el video
cv2.namedWindow('stream')
cv2.resizeWindow('stream', 960, 540)

# Se hace la detección cada 60 fotogramas
while(vid.isOpened()):
    perform_detection = count % 60 == 0
    ok, frame = vid.read()

    if ok:
        if perform_detection:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0,0,0], 1, crop=False)
            model.setInput(blob)
            outputs = model.forward(outputNames)
            bboxes, probs, class_ids = where_is_it(frame, outputs)
            
            if len(bboxes) > 0:
                mtracker = cv2.legacy.MultiTracker_create()
                idxs = cv2.dnn.NMSBoxes(bboxes, probs, P_THRESH, NMS_THRESH)
                idxs = np.array(idxs).flatten()
                for i in idxs:
                    bbox = [int(v) for v in bboxes[i]]
                    x, y, w, h = bbox
                    mtracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, (x, y, w, h))
                count += 1
            else:
                cv2.putText(frame, 'Detection failed', (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        else:
            is_tracking, bboxes = mtracker.update(frame)
            if is_tracking:
                for i, bbox in enumerate(bboxes):
                    x, y, w, h = [int(val) for val in bbox]
                    class_id = classes[class_ids[idxs[i]]]
                    col = [int(c) for c in colors[class_ids[idxs[i]], :]]
                    
# Marca el cuadro de seguimiento con el color correspondiente y escribe el nombre de la clase encima
                    cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
                    cv2.putText(frame, class_id, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                count += 1
                
# Si el seguimiento falla se reinicia el contador para activar la detección
            else:
                count = 0
        cv2.imshow('stream', frame)
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        print('Exhausted video capture.')
        break
out.release()
cv2.destroyAllWindows()
