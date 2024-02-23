import numpy as np
from typing import Tuple
import copy


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
