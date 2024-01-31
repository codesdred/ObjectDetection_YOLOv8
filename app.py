import cv2
from flask import Flask, render_template, Response
import argparse
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[480, 480],
        nargs=2,
        type=int
    )

    args = parser.parse_args()
    return args

def get_frame():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('yolov8n.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        result = model(frame, classes=0)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]

        total_objects_count = len(labels)
        counter_text = f"Total Objects: {total_objects_count}"

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.putText(frame, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
