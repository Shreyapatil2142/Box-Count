# ========================= FINAL + MONGODB + VISUAL TRACKING (GREEN = COUNTED) =========================
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import json
import os
import threading
import queue
import time
from datetime import date
from collections import deque
from pymongo import MongoClient

# ------------------- MONGODB ATLAS (FREE TIER) -------------------
# Replace with your actual MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://vishaljangid2004as:Vishal9767@vishal.tlkuemw.mongodb.net/?appName=Vishal"
DB_NAME = "warehouse"
COLLECTION_NAME = "daily_box_counts"

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

collection = get_db()

# Ensure today's record exists
today_str = date.today().isoformat()
collection.update_one({"date": today_str}, {"$setOnInsert": {"count": 0}}, upsert=True)

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Loading Bay - Smart Counter", layout="wide")
st.title("Smart Box Counter | Green = Counted | MongoDB Cloud")
st.success("Connected to MongoDB Atlas")

RTSP_URL = "rtsp://admin:Apple%409978@150.129.50.173:554/stream1"
CONF_THRESHOLD = 0.45
ALERT_LIMIT = 5

# ------------------- LOAD ROI -------------------
if not os.path.exists("roi.json"):
    st.error("roi.json not found!")
    st.stop()

with open("roi.json") as f:
    data = json.load(f)

pts = np.array(data, dtype=np.float32) if isinstance(data[0], list) else np.array(data, dtype=np.float32).reshape(-1, 2)
ROI_POLYGON = pts.reshape(-1, 1, 2).astype(np.float32)

def is_inside_roi(x1, y1, x2, y2):
    cx = float(x1 + x2) / 2
    cy = float(y1 + y2) / 2
    return cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0

# ------------------- MODEL -------------------
@st.cache_resource
def load_model():
    return YOLO("train2/weights/best.pt")

model = load_model()

# ------------------- TRACKING (VISUAL + NO DUPLICATES) -------------------
tracked_boxes = {}  # id → {"centers": deque, "counted": False}
MAX_HISTORY = 15
DIST_THRESHOLD = 100

# ------------------- UI -------------------
frame_ph = st.empty()
c1, c2, c3, c4 = st.columns(4)
current_ph = c1.empty()
today_ph = c2.empty()
fps_ph = c3.empty()
status_ph = c4.empty()
alert_ph = st.sidebar.empty()

# Get today's total from MongoDB
today_total = collection.find_one({"date": today_str})["count"]
today_ph.metric("Today's Total Boxes", today_total)

# Last 7 days
with st.expander("Last 7 Days Report"):
    history = collection.find().sort("date", -1).limit(7)
    for entry in history:
        st.write(f"**{entry['date']}** → {entry['count']} boxes")

# ------------------- CAPTURE THREAD -------------------
frame_queue = queue.Queue(maxsize=2)

def capture_thread():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    delay = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(delay)
            delay = min(delay * 2, 30)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        delay = 1
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)

threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=capture_thread, daemon=True).start()
time.sleep(3)

frame_count = 0
start_time = time.time()

while True:
    try:
        frame = frame_queue.get(timeout=1)
    except:
        status_ph.error("NO SIGNAL")
        continue

    frame_count += 1
    status_ph.success("LIVE • TRACKING")

    small = cv2.resize(frame, (640, 640))
    results = model(small, conf=CONF_THRESHOLD, verbose=False)[0]

    h, w = frame.shape[:2]
    sx = w / 640
    sy = h / 640

    annotated = frame.copy()
    current_in_roi = 0
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * [sx, sy, sx, sy]).astype(int)
        conf = box.conf.item()
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        detections.append({"bbox": (x1, y1, x2, y2), "center": center, "conf": conf})

    # === TRACKING LOGIC ===
    matched_ids = set()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        center = det["center"]
        inside = is_inside_roi(x1, y1, x2, y2)

        if inside:
            current_in_roi += 1

        # Match with existing tracked boxes
        matched = False
        for obj_id, data in list(tracked_boxes.items()):
            if obj_id in matched_ids: continue
            last_center = data["centers"][-1]
            dist = np.linalg.norm(np.array(center) - np.array(last_center))
            if dist < DIST_THRESHOLD:
                data["centers"].append(center)
                matched_ids.add(obj_id)
                matched = True

                # Mark as counted only once
                if inside and not data["counted"]:
                    collection.update_one(
                        {"date": today_str},
                        {"$inc": {"count": 1}}
                    )
                    data["counted"] = True
                    today_total += 1
                    today_ph.metric("Today's Total Boxes", today_total)

                # Draw: GREEN if counted, YELLOW if not
                color = (0, 255, 0) if data["counted"] else (0, 255, 255)
                thick = 7 if data["counted"] else 3
                label = f"ID:{obj_id} {'COUNTED' if data['counted'] else 'NEW'}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                cv2.putText(annotated, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                break

        # New box
        if not matched and inside:
            new_id = len(tracked_boxes)
            tracked_boxes[new_id] = {
                "centers": deque([center], maxlen=MAX_HISTORY),
                "counted": False
            }

    # Clean old tracks
    tracked_boxes = {k: v for k, v in tracked_boxes.items() if len(v["centers"]) > 0}

    # === DRAW ROI ===
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [ROI_POLYGON.astype(np.int32)], (255, 150, 50))
    cv2.polylines(overlay, [ROI_POLYGON.astype(np.int32)], True, (0, 0, 255), 10)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    first_pt = ROI_POLYGON[0][0]
    cv2.putText(annotated, "LOADING BAY", (int(first_pt[0]), int(first_pt[1])-60),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 8)

    # === UPDATE UI ===
    frame_ph.image(annotated, channels="BGR", use_container_width=True)
    current_ph.metric("Currently in Bay", current_in_roi)
    fps_ph.metric("FPS", f"{frame_count/(time.time()-start_time):.1f}")

    if current_in_roi > ALERT_LIMIT:
        alert_ph.error(f"OVER CAPACITY → {current_in_roi} BOXES!")
    else:
        alert_ph.empty()