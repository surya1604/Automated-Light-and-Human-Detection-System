import cv2
from flask import Flask, render_template, Response, request
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

detector = YOLO("cctv.pt")

isStreaming = True

# Variable to track the start time
start_time = time.time()

# Variable to store the frame count
frame_count = 0

# Function to detect persons in an image
def detect_persons(frame):
    person_detected = False
    bounding_boxes = []

    results = detector(frame)

    # Process results list
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    final_boxes = []

    for box, label, conf in zip(boxes, classes, confidences):
        if label == 1 and conf >= 0.50:
            final_boxes.append(box)
            person_detected = True

    return final_boxes, person_detected

# Function to draw bounding boxes on frame
def draw_boxes(frame, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
    return frame


global cap
executor = ThreadPoolExecutor(max_workers=1)
cap = cv2.VideoCapture(0)  

def gen_frames():
    global isStreaming, start_time, frame_count
    while True:
        if not isStreaming:
            continue

        ret, frame = cap.read()

        if not ret:
            break

        # Detect persons in the frame and get bounding boxes
        bounding_boxes, person_detected = detect_persons(frame)

        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame, bounding_boxes) if bounding_boxes else frame

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

        # Yield detection status
        yield "data: {}\n\n".format(person_detected)

        frame_count += 1

        # Calculate fps every 10 frames
        if frame_count % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps}")
            start_time = time.time()
            frame_count = 0


def detection_status():
    for status in gen_frames():
        yield status


@app.route('/')
def index():
    return render_template('trial3.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/task', methods=['POST','GET'])
# def tasks():
#     global isStreaming, cap
#     if request.form.get('stop') == 'Stop/Start':
#         if isStreaming:
#             cap.release()
#             cv2.destroyAllWindows()   
#             isStreaming = False   
#         else:
#             cap = cv2.VideoCapture(0)
#             isStreaming = True
#     elif request.method=='GET':
#         return render_template('trial3.html')
#     return render_template('trial3.html')

@app.route('/detection_status')
def detection_status_feed():
    return Response(detection_status(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

cap.release()
cv2.destroyAllWindows()
