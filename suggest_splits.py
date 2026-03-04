import cv2
import mediapipe as mp
import numpy as np
import json
import sys
import os

# Cấu hình
TARGET_PROCESS_FPS = 6  # Chỉ xử lý 6 khung hình trên mỗi giây video

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Giảm xuống 0 để chạy cực nhanh, 1 là cân bằng
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def is_resting_pose(landmarks, prev_landmarks=None):
    if not landmarks:
        return False

    # Get key landmarks
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Condition 1: Wrists below elbows and near hips
    wrist_low_left = l_wrist.y > l_elbow.y and abs(l_wrist.y - l_hip.y) < 0.2
    wrist_low_right = r_wrist.y > r_elbow.y and abs(r_wrist.y - r_hip.y) < 0.2

    if not (wrist_low_left and wrist_low_right):
        return False

    # Condition 2: Low motion
    if prev_landmarks:
        # Khoảng cách di chuyển của 2 cổ tay
        dist = np.linalg.norm(np.array([l_wrist.x, l_wrist.y]) - np.array(
            [prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])) + \
               np.linalg.norm(np.array([r_wrist.x, r_wrist.y]) - np.array(
                   [prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]))

        # Vì ta nhảy cóc frame nên threshold chuyển động cần tăng nhẹ một chút (0.05 -> 0.08)
        if dist > 0.08:
            return False

    return True


def suggest_splits(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps

    # Tính toán số lượng frame cần nhảy (ví dụ 30fps / 6fps = nhảy mỗi 5 frame)
    skip_step = max(1, int(video_fps / TARGET_PROCESS_FPS))

    sys.stderr.write(f"INFO: Video {video_fps} FPS. Processing every {skip_step} frames.\n")

    splits = []
    rest_start = None
    prev_landmarks = None

    for f_idx in range(0, frame_count, skip_step):
        # Nhảy tới frame cần xử lý
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Log tiến độ cho Electron (Dạng PROGRESS:10)
        percent = int((f_idx / frame_count) * 100)
        sys.stderr.write(f"PROGRESS:{percent}\n")
        sys.stderr.flush()

        # Xử lý Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        current_time = f_idx / video_fps
        is_rest = is_resting_pose(landmarks, prev_landmarks)

        if is_rest:
            if rest_start is None:
                rest_start = current_time
        else:
            if rest_start is not None:
                rest_duration = current_time - rest_start
                if rest_duration > 0.6:  # Nếu nghỉ > 0.6s thì đánh dấu điểm cắt
                    split_time = rest_start + rest_duration / 2
                    splits.append(round(split_time, 3))
                rest_start = None

        prev_landmarks = landmarks

    # Kết thúc
    if rest_start is not None and (duration - rest_start) > 0.5:
        splits.append(round(rest_start + (duration - rest_start) / 2, 3))

    cap.release()
    sys.stderr.write("PROGRESS:100\n")
    return {
        "splits": list(dict.fromkeys(splits)),  # Xóa trùng nếu có
        "fps": video_fps,
        "duration": duration
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No path"}))
        sys.exit(1)

    video_path = sys.argv[1]
    result = suggest_splits(video_path)
    print(json.dumps(result))