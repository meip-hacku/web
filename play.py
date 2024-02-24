from flask import Blueprint, render_template, jsonify
import cv2
import tensorflow as tf
import numpy as np
import base64
import time
import analysis
import openai
import os
from dotenv import load_dotenv
from flask_socketio import SocketIO
import torch
import random


def set_socket(s: SocketIO):
    global socket
    socket = s


def set_model(m):
    global model
    model = m


CSV_HEADER = [
    'nose(x)', 'nose(y)', 'left eye(x)', 'left eye(y)',
    'right eye(x)', 'right eye(y)', 'left ear(x)', 'left ear(y)',
    'right ear(x)', 'right ear(y)', 'left shoulder(x)', 'left shoulder(y)',
    'right shoulder(x)', 'right shoulder(y)', 'left elbow(x)', 'left elbow(y)',
    'right elbow(x)', 'right elbow(y)', 'left wrist(x)', 'left wrist(y)',
    'right wrist(x)', 'right wrist(y)', 'left hip(x)', 'left hip(y)',
    'right hip(x)', 'right hip(y)', 'left knee(x)', 'left knee(y)',
    'right knee(x)', 'right knee(y)', 'left ankle(x)', 'left ankle(y)',
    'right ankle(x)', 'right ankle(y)'
]


class InputData:
    def __init__(self):
        self.data = np.zeros((0, 34)).astype(np.float32)
        self.t = []
        self.last_time = 0
        self.last_splitted = 0.0
        self.splitted = []
        self.splitted_index = 0
        self.squat_flag = False
        self.squat_count = 0
        self.standing_pose = None
        self.first = True

    def add(self, posedata) -> bool:
        # スクワットをしているフラグを返却
        if self.first:
            if posedata.score < 0.4:
                return False
            if analysis.estimate_standing(posedata):
                print("Standing!!")
                self.data = np.array(posedata.all).reshape(1, 34)
                self.standing_pose = posedata
                print(self.standing_pose.nose, self.standing_pose.rightAnkle, self.standing_pose.leftAnkle)
                self.t = [0]
                self.last_time = time.time()
                self.last_splitted = time.time()
                self.splitted_index = 0
                self.first = False
            return False

        self.data = np.append(self.data, np.array(posedata.all).reshape(1, 34), axis=0)
        # 経過した時間を記録しておく
        t = time.time()
        self.t.append(t - self.last_time)
        self.last_time = t

        # 立っているか
        if analysis.estimate_squatting(posedata, self.standing_pose):
            if not self.squat_flag:
                self.squat_flag = True
        elif analysis.estimate_standing(posedata) and t - self.last_splitted > 0.7:
            if self.squat_flag:
                self.squat_flag = False
                self.squat_count += 1
                self.splitted.append(self.data[self.splitted_index:])
                self.splitted_index = len(self.data)
                self.last_splitted = t
                print(self.squat_count)
                return True
        return False

    def __getitem__(self, header: str) -> np.ndarray:
        i = self._header(header)
        return self.data[:, i]

    def __setitem__(self, header: str, value: np.ndarray):
        i = self._header(header)
        self.data[:, i] = value

    def _header(self, header: str) -> int:
        assert header in CSV_HEADER, f"Invalid header: {header}"
        index = CSV_HEADER.index(header)
        return index

    def get_splitted_data(self):
        splitted_data = []
        for s in self.splitted[:5]:
            if s.shape[0] < MAX_SEQ_LEN:
                padding = np.zeros((MAX_SEQ_LEN - s.shape[0], 34))
                splitted_data.append(np.append(s, padding, axis=0))
            else:
                splitted_data.append(s[-MAX_SEQ_LEN:])
        return np.array(splitted_data).astype(np.float32)


class PoseData:
    def __init__(self, data, embedding_info):
        self.data = data
        self.embedding_info = embedding_info
        self.transform()
        self.nose = (data[0][1], data[0][0])
        self.leftEye = (data[1][1], data[1][0])
        self.rightEye = (data[2][1], data[2][0])
        self.leftEar = (data[3][1], data[3][0])
        self.rightEar = (data[4][1], data[4][0])
        self.leftShoulder = (data[5][1], data[5][0])
        self.rightShoulder = (data[6][1], data[6][0])
        self.leftElbow = (data[7][1], data[7][0])
        self.rightElbow = (data[8][1], data[8][0])
        self.leftWrist = (data[9][1], data[9][0])
        self.rightWrist = (data[10][1], data[10][0])
        self.leftHip = (data[11][1], data[11][0])
        self.rightHip = (data[12][1], data[12][0])
        self.leftKnee = (data[13][1], data[13][0])
        self.rightKnee = (data[14][1], data[14][0])
        self.leftAnkle = (data[15][1], data[15][0])
        self.rightAnkle = (data[16][1], data[16][0])

        self.score = 0.0
        for i in range(17):
            self.score += data[i][2]
        self.score /= 17.0

        self.all = [data[i][j] for j in range(1, -1, -1) for i in range(17)]
        self.probs = [data[i][2] for i in range(17)]

    def transform(self):
        self.data = np.array(self.data)
        self.data[:, 0] = self.data[:, 0] * self.embedding_info["target_w"] - self.embedding_info["x"]
        self.data[:, 1] = self.data[:, 1] * self.embedding_info["target_h"] - self.embedding_info["y"]


play_bp = Blueprint('play_bp', __name__)
model = 'static/models/movenet_lightning.tflite'
interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
starting = False
socket: SocketIO = None
model: torch.nn.Module = None
input_data = InputData()
MAX_SEQ_LEN = 133


def resize_keep_aspect(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect = w / h
    target_aspect = target_w / target_h
    if aspect > target_aspect:
        # calculate new image width and height
        new_w = int(target_w)
        new_h = int(target_w / aspect)
    else:
        new_w = int(target_h * aspect)
        new_h = int(target_h)
    # resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    # create new image of desired size and color (black) for padding
    new_image = np.zeros((target_h, target_w, 3), np.uint8)
    # calculate x,y position where image should be placed
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    # copy resized image to new image
    new_image[y:y + new_h, x:x + new_w] = resized_image
    embedding_info = {
        "x": x,
        "y": y,
        "new_w": new_w,
        "new_h": new_h,
        "target_w": target_w,
        "target_h": target_h,
        "aspect": aspect
    }
    return new_image, embedding_info


def infer(image):
    image_resized, embedding_info = resize_keep_aspect(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    interpreter.set_tensor(input_details[0]['index'], [image_resized.astype('uint8')])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return PoseData(output_data[0][0], embedding_info) if output_data.any() else None


@play_bp.route('/play')
def play():
    global input_data, starting
    starting = False
    input_data.__init__()
    return render_template('play.html')


def frame(data):
    global starting, input_data
    if not starting:
        return
    sbuf = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(sbuf, np.uint8)
    if nparr.shape[0] == 0:
        return
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    try:
        posedata = infer(frame)
    except Exception as e:
        print(e)
        return
    if posedata:
        squat_flag = input_data.add(posedata)
        if squat_flag:
            socket.emit("squat", input_data.squat_count)
            if input_data.squat_count == 2:
                input_data.squat_count = 0
                socket.emit('finish', "")
                starting = False


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def get_score(_input_data):
    model.eval()
    data = _input_data.get_splitted_data()
    data = analysis.extract_features(data)
    lengths = [d.shape[0] for d in data]
    data = torch.tensor(data)
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=MAX_SEQ_LEN)
    result = model(data, padding_masks).detach().numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    result = sigmoid(result)
    result = np.log(1 + result) / np.log(2)
    result = np.mean(result, axis=0)  # (5, 4) -> (4,)
    for i in range(4):
        r = result[i]
        if r < 0.0:
            result[i] = random.random() * 0.3 + 0.6
    return (result * 100).tolist()


@play_bp.route('/asyncResult', methods=['POST'])
def async_result():
    global input_data
    time.sleep(3)
    score = get_score(input_data)
    print(score)
    comment = get_comment(score)
    return jsonify({
        "score": score,
        "comment": comment
    })


def get_comment(score):
    load_dotenv()
    openai.api_key = os.environ.get('OPENAI_APIKEY')

    prompt = f'''
あなたは筋トレトレーナーです。とあるトレーニーがスクワットを行ったところ、
「膝の位置」が{int(score[0])}/100点,
「腰を落とせているか」が{int(score[1])}/100点,
「上体の起こし具合がちょうどいいか」が{int(score[2])}/100点,
「きちんと胸を張れているかどうか」が{int(score[3])}/100点
でした。この結果に対してフィードバックコメントをしてください。
あなたの自己紹介は不要です。点数には言及しないでください。300文字程度で答えなさい。
'''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.0,
    )

    comment = response['choices'][0]['message']['content']
    print(comment)
    return comment


@play_bp.route('/result')
def result():
    # print(input_data.data)
    # score = get_score(input_data)
    # print(score)
    return render_template('result.html')


@play_bp.route('/start', methods=['POST'])
def start():
    global starting
    starting = True
    return "start"
