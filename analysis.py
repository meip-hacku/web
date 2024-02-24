import numpy as np
from typing import Tuple
import copy

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


def normalize_data(data, standing_pose) -> np.ndarray:
    """
    鼻のy座標の最小値が0.0，足のy座標の最大値が1.0になるように正規化
    鼻のx座標の平均が0.0になるようにし，アスペクト比を保って正規化
    """

    nose_y_min = standing_pose.nose[1]
    ankle_y_max = np.max((data.leftAnkle[1], data.rightAnkle[1]))

    nose_x_mean = standing_pose.nose[0]

    normalized = copy.deepcopy(data.all)

    for i, d in enumerate(data.all):
        if i % 2 == 0:
            normalized[i] = (d - nose_x_mean) / (ankle_y_max - nose_y_min)
        else:
            normalized[i] = (d - nose_y_min) / (ankle_y_max - nose_y_min)

    return normalized


def angle(center: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    3点の角度を求める
    """
    a = np.array(a) - np.array(center)
    b = np.array(b) - np.array(center)
    dot = a[0] * b[0] + a[1] * b[1]
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cos = dot / norm
    return np.arccos(cos) * 180 / np.pi


def estimate_standing(data) -> bool:
    """
    立っているかどうかを推定する
    """

    def check_knee_angle(left: float, right: float) -> bool:
        if right > 165 and left > 120:
            return True
        if right > 120 and left > 165:
            return True
        return False

    def check_hip_angle(left: float, right: float) -> bool:
        if right > 165 and left > 120:
            return True
        if right > 120 and left > 165:
            return True
        return False

    # 膝の角度
    right_knee_angle = angle(data.rightKnee, data.rightHip, data.rightAnkle)
    left_knee_angle = angle(data.leftKnee, data.leftHip, data.leftAnkle)

    # 尻の角度
    right_hip_angle = angle(data.rightHip, data.rightKnee, data.rightShoulder)
    left_hip_angle = angle(data.leftHip, data.leftKnee, data.leftShoulder)

    # 膝の角度のどちらかが165度以上で尻の角度も165度以上なら立っていると判定
    if check_knee_angle(left_knee_angle, right_knee_angle) and check_hip_angle(left_hip_angle, right_hip_angle):
        return True

    return False


def estimate_squatting(data, standing_pose) -> bool:
    """
    腰を落としているかどうかを推定する
    """
    normalized = normalize_data(data, standing_pose)
    left_hip_y = normalized[23]
    right_hip_y = normalized[25]

    # 膝の角度
    right_knee_angle = angle(data.rightKnee, data.rightHip, data.rightAnkle)
    left_knee_angle = angle(data.leftKnee, data.leftHip, data.leftAnkle)

    def check_knee_angle(left: float, right: float) -> bool:
        if right < 80 or left < 80:
            return False
        if right < 120 and left > 150:
            return False
        if right > 150 and left < 120:
            return False
        if right < 120 or left < 120:
            return True
        return False

    # 膝の角度のどちらかが120度以下で鼻のy座標が0.1以下なら腰を落としていると判定
    if check_knee_angle(left_knee_angle, right_knee_angle) and (left_hip_y > 0.1 or right_hip_y > 0.1):
        return True
    return False


def angle_from_label(data, center_label: str, label_1: str, label_2: str) -> np.ndarray:
    x1_idx = CSV_HEADER.index(f"{label_1}(x)")
    y1_idx = CSV_HEADER.index(f"{label_1}(y)")
    x2_idx = CSV_HEADER.index(f"{label_2}(x)")
    y2_idx = CSV_HEADER.index(f"{label_2}(y)")
    x3_idx = CSV_HEADER.index(f"{center_label}(x)")
    y3_idx = CSV_HEADER.index(f"{center_label}(y)")

    x1 = data[:, x1_idx] - data[:, x3_idx]
    y1 = data[:, y1_idx] - data[:, y3_idx]
    x2 = data[:, x2_idx] - data[:, x3_idx]
    y2 = data[:, y2_idx] - data[:, y3_idx]

    dot_product = x1 * x2 + y1 * y2
    magnitude_1 = np.sqrt(x1 ** 2 + y1 ** 2)
    magnitude_2 = np.sqrt(x2 ** 2 + y2 ** 2)

    cos_theta = dot_product / (magnitude_1 * magnitude_2)
    angle = np.arccos(cos_theta)

    if np.isnan(angle).any():
        return np.zeros_like(angle)
    return angle


def extract_features(_data) -> np.ndarray:
    new_data = []
    for batch in range(_data.shape[0]):
        data = _data[batch]
        nose_y = data[:, CSV_HEADER.index("nose(y)")]

        nose_left_shoulder_hip_angle = angle_from_label(data, "nose", "left shoulder", "left hip")
        nose_right_shoulder_hip_angle = angle_from_label(data, "nose", "right shoulder", "right hip")

        nose_left_hip_knee_angle = angle_from_label(data, "nose", "left hip", "left knee")
        nose_right_hip_knee_angle = angle_from_label(data, "nose", "right hip", "right knee")

        left_shoulder_hip_knee_angle = angle_from_label(data, "left shoulder", "left hip", "left knee")
        right_shoulder_hip_knee_angle = angle_from_label(data, "right shoulder", "right hip", "right knee")

        left_hip_knee_ankle_angle = angle_from_label(data, "left hip", "left knee", "left ankle")
        right_hip_knee_ankle_angle = angle_from_label(data, "right hip", "right knee", "right ankle")

        left_hip_x = data[:, CSV_HEADER.index("left hip(x)")]
        right_hip_x = data[:, CSV_HEADER.index("right hip(x)")]
        left_hip_y = data[:, CSV_HEADER.index("left hip(y)")]
        right_hip_y = data[:, CSV_HEADER.index("right hip(y)")]

        left_knee_x = data[:, CSV_HEADER.index("left knee(x)")]
        right_knee_x = data[:, CSV_HEADER.index("right knee(x)")]
        left_knee_y = data[:, CSV_HEADER.index("left knee(y)")]
        right_knee_y = data[:, CSV_HEADER.index("right knee(y)")]

        new_data.append(np.array([
            nose_y,
            nose_left_shoulder_hip_angle,
            nose_right_shoulder_hip_angle,
            nose_left_hip_knee_angle,
            nose_right_hip_knee_angle,
            left_shoulder_hip_knee_angle,
            right_shoulder_hip_knee_angle,
            left_hip_knee_ankle_angle,
            right_hip_knee_ankle_angle,
            left_hip_x,
            right_hip_x,
            left_hip_y,
            right_hip_y,
            left_knee_x,
            right_knee_x,
            left_knee_y,
            right_knee_y
        ]).T)

    return np.array(new_data)
