from flask import Blueprint, render_template
import cv2
import tensorflow as tf
from app import socketio
import numpy as np
import base64

play_bp = Blueprint('play_bp', __name__)
model = 'static/models/movenet_lightning.tflite'
interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class PoseData:
    def __init__(self, data):
        self.data = data
        self.nose = (data[0][0], data[0][1])
        self.leftEye = (data[1][0], data[1][1])
        self.rightEye = (data[2][0], data[2][1])
        self.leftEar = (data[3][0], data[3][1])
        self.rightEar = (data[4][0], data[4][1])
        self.leftShoulder = (data[5][0], data[5][1])
        self.rightShoulder = (data[6][0], data[6][1])
        self.leftElbow = (data[7][0], data[7][1])
        self.rightElbow = (data[8][0], data[8][1])
        self.leftWrist = (data[9][0], data[9][1])
        self.rightWrist = (data[10][0], data[10][1])
        self.leftHip = (data[11][0], data[11][1])
        self.rightHip = (data[12][0], data[12][1])
        self.leftKnee = (data[13][0], data[13][1])
        self.rightKnee = (data[14][0], data[14][1])
        self.leftAnkle = (data[15][0], data[15][1])
        self.rightAnkle = (data[16][0], data[16][1])

def draw_dots(image, posedata):
    dot_color = (0, 0, 255)
    dot_radius = 10
    height, width = image.shape[:2]
    for loc in posedata.data:
        x, y = int(loc[1] * width), int(loc[0] * height)
        cv2.circle(image, (x, y), dot_radius, dot_color, -1)
    return image

def infer(image):
    image_resized = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    interpreter.set_tensor(input_details[0]['index'], [image_resized.astype('uint8')])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return PoseData(output_data[0][0])


@play_bp.route('/play')
def play():
    return render_template('play.html')

@play_bp.route('/result')
def result():
    return render_template('result.html', result=[20, 30, 40, 50, "comment, comment, comment"])

def frame(data):
    sbuf = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(sbuf, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output = infer(frame)
    # frame = draw_dots(frame, output)
    # ret, buffer = cv2.imencode('.jpg', frame)
    # frame = buffer.tobytes()