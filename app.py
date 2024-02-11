from flask import Flask, render_template, Response
from inference_classifier import GestureClassifier
import cv2

app = Flask(__name__)
gesture_classifier = GestureClassifier()
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform gesture classification using your GestureClassifier
        predicted_character, frame = gesture_classifier.predict(frame)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
