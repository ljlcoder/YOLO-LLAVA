print(f"Loading..")
import time
import os
import cv2
import queue
import threading
import numpy as np
import math
from datetime import datetime
from ultralytics import YOLO
import torch
import torchvision
import base64
import requests
import json

model = "yolo11s"
# rtsp_stream = "rtsp://user@password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
ollama = "http://127.0.0.1:11434/api/generate"

rtsp_stream = "GH010076.MP4"
labels = open("coco.names").read().strip().split("\n")
buffer = 512
idle_reset = 3000
min_confidence = 0.15
min_size = 20
class_confidence = {
    "truck": 0.35,
    "car": 0.25,
    "boat": 0.85,
    "bus": 0.5,
    "aeroplane": 0.85,
    "frisbee": 0.88,
    "pottedplant": 0.55,
    "train": 0.85,
    "chair": 0.5,
    "parking meter": 0.9,
    "fire hydrant": 0.65
}

prompts = {
    "person": "get gender and age of this person in 5 words or less",
    "car": "get body type and color of this car in 5 words or less"
}
snapshot_directory = "snapshots"

frames = 0
prev_frames = 0
last_frame = 0
fps = 0
WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0
recording = False
out = None

opsize = (640, 480)
streamsize = (0, 0)


def preinit():
    for folder in ["elements", "models", "recordings", "snapshots"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"-- created folder: {folder}")


def transform(xmin, ymin, xmax, ymax):
    x_scale = streamsize[0] / opsize[0]
    y_scale = streamsize[1] / opsize[1]

    new_xmin = int(xmin * x_scale)
    new_ymin = int(ymin * y_scale)
    new_xmax = int(xmax * x_scale)
    new_ymax = int(ymax * y_scale)
    return (new_xmin, new_ymin, new_xmax, new_ymax)


def resample(frame):
    return cv2.resize(frame, opsize, interpolation=cv2.INTER_AREA)


def rest(url, payload):
    headers = {'Content-Type': 'application/json'}
    r = False
    try:
        data = data = json.dumps(payload)
        response = requests.post(url, data, headers=headers)
        if (response.status_code == 200):
            r = json.loads(response.text)
        else:
            print(response.text)
            return False
    except Exception as e:
        print(f"-- error {e}")
    finally:
        return r


def millis():
    return round(time.perf_counter() * 1000)


def timestamp():
    return int(time.time())


object_count = 0
old_count = 0
obj_break = millis()
obj_idle = 0
obj_list = []
obj_max = 16
obj_avg = 0
fskip = False
last_fskip = timestamp()
app_start = timestamp()
obj_score = labels

bounding_boxes = []
point_timeout = 8000
stationary_val = 16

obj_number = 1


def crc32(string):
    crc = 0xFFFFFFFF
    for char in string:
        byte = ord(char)
        for _ in range(8):
            if (crc ^ byte) & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
            byte >>= 1
    return crc ^ 0xFFFFFFFF


def genprompt(t):
    if t in prompts:
        return prompts[t]
    return "describe this image in 5 words or less"


def center(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    return (center_x, center_y)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _size(x1, y1, x2, y2):
    return abs(x1 - y2)


def bearing(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    bearing_rad = math.atan2(delta_y, delta_x)
    bearing_deg = math.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def direction(bearing):
    normalized_bearing = bearing % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(normalized_bearing / 45) % 8
    return directions[index]


def similar(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def match(img1, img2):
    max_val = 0
    try:
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
    except Exception as e:
        max_val = similar(img1, img2)
        # print(f"An error occurred: {e}")
    finally:
        return max_val


class BoundingBox:
    def __init__(self, name, points, size, image, buffer=stationary_val):
        global obj_number
        self.nr = obj_number
        obj_number += 1
        self.x, self.y = points
        self.px = 0
        self.py = 0
        self.buffer = buffer
        self.created = millis()
        self.timestamp = self.created
        self.size = size
        self.sid = str(crc32(f'{self.x}-{self.y}-{self.timestamp}-{self.size}'))
        self.name = name
        self.checkin = True
        self.detections = 0
        self.distance = 0
        self.idle = 0
        self.image = image
        self.desc = False
        self.state = 0
        self.seen = self.created

        self.init()
        print("New object: " + self.name + "#" + str(self.nr) + " size:" + str(self.size))
        self.save("elements/" + self.name + "-" + str(self.nr) + ".png")

    def see(self):
        self.seen = millis()

    def ping(self):
        self.timestamp = millis()
        idle = self.timestamp - self.created
        if (idle >= 1000):
            self.idle = idle // 1000
        else:
            self.idle = 0
        return self.idle

    def save(self, file):
        cv2.imwrite(file, self.image)

    def export(self):
        _, buffer = cv2.imencode('.png', self.image)
        base64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return base64_image

    def init(self):
        self.min_x = self.x - self.buffer
        self.max_x = self.x + self.buffer
        self.min_y = self.y - (self.buffer)
        self.max_y = self.y + (self.buffer)

    def contains(self, x, y):
        return ((self.checkin == False) and self.min_x <= x <= self.max_x) and (self.min_y <= y <= self.max_y)

    def update(self, time, new_x, new_y):
        self.checkin = True
        self.timestamp = time
        idle = self.timestamp - self.created
        if (idle >= 1000):
            self.idle = idle // 1000
        else:
            self.idle = 0
        self.px = self.x
        self.py = self.y
        self.x = new_x
        self.y = new_y
        self.detections += 1
        self.init()

    def update_in_array(self, time, new_x, new_y, bounding_boxes):
        for bbox in bounding_boxes:
            if bbox.sid == self.sid:
                bbox.update(time, new_x, new_y)
                return True
            return False


def resetIteration():
    global bounding_boxes
    [setattr(item, 'checkin', False) for item in bounding_boxes]


def closest(bounding_boxes, reference_point, class_name, size):
    closest_bbox = False
    min_distance = float('inf')

    for bbox in bounding_boxes:
        if (bbox.idle >= 3 or abs(bbox.size - size) > 10):
            continue

        dx = bbox.x - reference_point[0]
        dy = bbox.y - reference_point[1]
        distance = math.sqrt(dx * dx + dy * dy)
        if (distance < 3 or distance > 200):
            continue

        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox

    if (closest_bbox != False and distance > 0):
        closest_bbox.distance = distance
        closest_bbox.update(millis(), reference_point[0], reference_point[1])
    return closest_bbox


def blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(mag)


def findSimilar(ref):
    closest_bbox = False
    score = 0.85
    for bbox in bounding_boxes:
        s = similar(ref, bbox.image)
        if (s > score):
            score = s
            closest_bbox = bbox
    # print("similar:"+str(score))
    return closest_bbox


def findMatch(ref):
    closest_bbox = False
    score = 0.96
    for bbox in bounding_boxes:
        s = match(ref, bbox.image)
        if (s > score):
            score = s
            closest_bbox = bbox
    # print("score:"+str(score))
    return closest_bbox


def closestEx(bounding_boxes, reference_point, class_name, size):
    return False

    point = reference_point
    found = []
    for i in range(6):
        c = closest(bounding_boxes, point, class_name, size)
        if (c == False and i == 0):
            return False
        if (c == False and i > 0):
            return found[-1]
        if (i == 1 and found[-1].sid == c.sid):
            return c

        point = (c.x, c.y)
        found.append(c)
        print("iteration " + str(i))

    return found[-1]


def getObject(point, cname):
    global bounding_boxes
    x, y = point
    time = millis()

    i = 0
    while i < len(bounding_boxes):
        bbox = bounding_boxes[i]
        if (cname != bbox.name):
            i += 1
            continue

        if bbox.contains(x, y):
            bbox.update(time, x, y)
            return bbox
        if (time - bbox.seen) >= point_timeout:
            del bounding_boxes[i]
        else:
            i += 1

    return False


def take_snapshot(frame):
    global snapshot_directory

    if not os.path.exists(snapshot_directory):
        os.makedirs(snapshot_directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(snapshot_directory, filename)

    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved: {filepath}")


def start_recording(cap):
    global recording, out
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20, (640, 480))

        recording = True
        print(f"Started recording: {filename}")


def stop_recording():
    global recording, out
    if recording:
        out.release()
        recording = False
        print("Stopped recording")


def add(num):
    if len(obj_list) >= obj_max:
        obj_list.pop(0)
    obj_list.append(num)


def average():
    l = len(obj_list)
    if (l <= 0):
        return 0
    return round(sum(obj_list) / l)


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=8):
    def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length):
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = np.array([int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                              int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)])
            end = np.array([int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
                            int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)])
            cv2.line(img, tuple(start), tuple(end), color, thickness)

    draw_dashed_line(img, pt1, (pt2[0], pt1[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt2[0], pt1[1]), pt2, color, thickness, dash_length)
    draw_dashed_line(img, pt2, (pt1[0], pt2[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt1[0], pt2[1]), pt1, color, thickness, dash_length)

    return img


def generate_color_shades(num_classes):
    colors = np.zeros((num_classes, 3), dtype=np.uint8)

    green = [0, 200, 0]
    orange = [0, 165, 255]
    yellow = [0, 200, 255]
    red = [0, 0, 255]

    base_colors = [green, orange, yellow, red]
    num_base_colors = len(base_colors)

    for i in range(num_classes):
        base_color_index = i % num_base_colors
        base_color = np.array(base_colors[base_color_index])
        shade_factor = (i // num_base_colors) / (num_classes // num_base_colors + 1)
        shade = base_color * (1 - shade_factor) + np.array([128, 128, 128]) * shade_factor
        colors[i] = shade.astype(np.uint8)
    return colors


print(f"Starting up..")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

colors = generate_color_shades(len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing model..")
model = YOLO("models/" + model + ".pt")
print(f"Loading model to {device}")
model.to(device)

loop = True
cap = cv2.VideoCapture(rtsp_stream)

if (rtsp_stream == 0):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = cap.get(cv2.CAP_PROP_FPS)
ret, img = cap.read()

streamsize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cv2.namedWindow(str(rtsp_stream), cv2.WINDOW_NORMAL)
cv2.resizeWindow(str(rtsp_stream), opsize[0], opsize[1])
cv2.setWindowProperty(str(rtsp_stream), cv2.WND_PROP_TOPMOST, 1)
q = queue.Queue(maxsize=buffer)


def stream():
    global cap, obj_idle, last_fskip, idle
    if cap.isOpened():
        ret, frame = cap.read()
        while loop:
            ret, frame = cap.read()
            if ret:
                if ((obj_idle > 0) and obj_idle >= idle_reset and (timestamp() - last_fskip >= 30)):
                    last_fskip = timestamp()
                    q.queue.clear()
                    obj_idle = 0
                    fskip = True
                    print(f"Frame skip")
                else:
                    q.put(frame)
            else:
                print("Can't receive frame (stream end?). Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def process(img):
    photo = img.copy()
    img = resample(img)
    global obj_score, bounding_boxes

    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        results = model(img_tensor, verbose=False, stream=True)

    obj_score = [0 for _ in range(len(obj_score))]
    c = 0;

    boxes = [box for r in results for box in r.boxes]

    now = millis()
    resetIteration()

    for box in boxes:
        class_id = int(box.cls)
        class_name = labels[class_id]
        confidence = float(box.conf)
        c = c + 1
        if ((class_name in class_confidence) and (confidence <= class_confidence[class_name])):
            continue

        if (confidence <= min_confidence):
            continue

        xmin, ymin, xmax, ymax = box.xyxy[0]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        width = xmax - xmin
        height = ymax - ymin

        if (xmin == 0 or ymin == 0 or xmax == 0 or ymax == 0 or xmax == img.shape[1] or ymax == img.shape[0]):
            continue

        if (class_name == "car" and (
                (width > height and (width / height) >= 2) or (width < min_size or height < min_size))):
            continue

        """
        color = colors[class_id].tolist()
        alpha = 0.35
        color_with_alpha = color + [alpha]
                 
        text = f"{class_name}"+" "+str(round(confidence, 6))         
        text_offset_x = xmin
        text_offset_y = ymin - 5
           
        overlay = img[ymin:ymax+1, xmin:xmax+1].copy()
        cv2.rectangle(overlay, (0, 0), (xmax-xmin, ymax-ymin), color_with_alpha, thickness=-1)
        cv2.addWeighted(overlay, alpha, img[ymin:ymax+1, xmin:xmax+1], 1 - alpha, 0, img[ymin:ymax+1, xmin:xmax+1])
        draw_dashed_rectangle(img,(xmin, ymin),(xmax, ymax),color,1,8)
       
        cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        continue
        """

        idx = class_id
        obj_score[idx] = obj_score[idx] + 1

        point = center(xmin, ymin, xmax, ymax)
        size = _size(xmin, ymin, xmax, ymax)

        obj = getObject(point, class_name);
        if (obj != False):
            obj.see()
            if (obj.desc != False):
                sid = obj.desc
            else:
                sid = obj.name + "#" + str(obj.nr)

            color = colors[class_id].tolist()
            cv2.circle(img, point, 1, (0, 0, 255), 2)

            cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            idle = str(obj.idle) + "s"
            cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        else:
            obj = closestEx(bounding_boxes, point, class_name, size)
            # obj = findMatch(snap)

            if (obj != False):
                print("picked up " + str(obj.nr) + "#" + obj.name + " from " + str(obj.distance))
                # print("found visual match: "+obj.name+"#"+str(obj.nr))
                cv2.line(img, point, (obj.x, obj.y), (0, 255, 255), 4)
                obj.see()

                if (obj.desc != False):
                    sid = obj.desc
                else:
                    sid = obj.name + "#" + str(obj.nr)

                cv2.circle(img, point, 1, (0, 255, 0), 2)

                cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                idle = str(obj.idle) + "s"
                cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            else:
                text = f"{class_name}" + " " + str(round(confidence, 6))
                text_offset_x = xmin
                text_offset_y = ymin - 5
                """
                color = colors[class_id].tolist()
                alpha = 0.35
                color_with_alpha = color + [alpha]
                       
                overlay = img[ymin:ymax+1, xmin:xmax+1].copy()
                cv2.rectangle(overlay, (0, 0), (xmax-xmin, ymax-ymin), color_with_alpha, thickness=-1)
                cv2.addWeighted(overlay, alpha, img[ymin:ymax+1, xmin:xmax+1], 1 - alpha, 0, img[ymin:ymax+1, xmin:xmax+1])
                draw_dashed_rectangle(img,(xmin, ymin),(xmax, ymax),color,1,8)
             
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                """
                cv2.circle(img, point, 1, (255, 255, 0), 2)
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                            1)

                qxmin, qymin, qxmax, qymax = transform(xmin, ymin, xmax, ymax)
                snap = photo[qymin:qymax, qxmin:qxmax]
                item = BoundingBox(class_name, point, size, snap)
                bounding_boxes.append(item)

    for obj in bounding_boxes:
        if (obj.checkin == False and obj.detections >= 3 and obj.idle > 0):
            obj.ping()
            if (now - obj.seen > 3000):
                continue

            if (obj.desc != False):
                sid = obj.desc
            else:
                sid = obj.name + "#" + str(obj.nr)

            idle = str(obj.idle) + "s"

            cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(img, sid, (obj.x, obj.y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.circle(img, (obj.x, obj.y), 1, (0, 255, 255), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    add(c);
    return img


def postreview():
    global bounding_boxes, loop
    while loop:
        for box in bounding_boxes:
            if box.state == 0:
                res = rest(ollama, {
                    "model": "llava",
                    "prompt": genprompt(box.name),
                    "images": [box.export()],
                    "stream": False
                })

                if res != False:
                    box.desc = res["response"].strip()
                    box.state = 1
        time.sleep(0.1)


bthread = threading.Thread(target=postreview)
bthread.start()

sthread = threading.Thread(target=stream)
sthread.start()

while loop:
    if ((q.empty() != True) and (fskip != True)):
        img = q.get_nowait()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            loop = False
        elif key == ord('r'):
            if not recording:
                start_recording(cap)
            else:
                stop_recording()
        elif key == ord('s'):
            take_snapshot(img)

        img = process(img)

        object_count = average()
        if object_count != old_count:
            obj_break = millis()
            obj_idle = 0
        else:
            obj_idle = millis() - obj_break

        cv2.putText(img, "Objects: " + str(object_count), (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, "Objects: " + str(object_count), (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        old_count = object_count;
        frames += 1
        if (millis() - last_frame >= 250):
            fps = (frames - prev_frames) * 4
            prev_frames = frames
            last_frame = millis()

        _fps = "FPS: " + str(fps)
        text_size = cv2.getTextSize(_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = 16
        text_y = img.shape[0] - 5
        cv2.putText(img, _fps, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, _fps, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        line = 16
        _t = line * 2
        for i, s in enumerate(obj_score):
            if (s > 0):
                _s = labels[i] + ": " + str(s)
                cv2.putText(img, _s, (16, _t), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(img, _s, (16, _t), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                _t = _t + line

        if recording:
            out.write(img)
            cv2.putText(img, "REC", (16, img.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, "REC", (16, img.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        bb = str(len(bounding_boxes))
        cv2.putText(img, "Tracking: " + bb, (16, img.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, "Tracking: " + bb, (16, img.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (64, 255, 255), 1)

        clock = datetime.now().strftime("%H:%M:%S")
        text_size = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = img.shape[1] - text_size[0] - 10
        text_y = img.shape[0] - 8
        cv2.putText(img, clock, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, clock, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow(str(rtsp_stream), img)

    else:
        fskip = False
        time.sleep(0.001)

bthread.join()
sthread.join()

cv2.destroyAllWindows()
print("Terminating..")
