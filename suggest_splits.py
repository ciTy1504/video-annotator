import cv2
import mediapipe as mp
import json
import sys
import os

cv2.setNumThreads(os.cpu_count())

# CONFIG
TARGET_PROCESS_FPS = 16
RESIZE_WIDTH = 320

# landmark index (tránh enum lookup)
L_WRIST = 15
R_WRIST = 16
L_ELBOW = 13
R_ELBOW = 14
L_HIP = 23
R_HIP = 24

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def is_resting_pose(landmarks, prev_landmarks=None):
    if not landmarks:
        return False

    l_wrist = landmarks[L_WRIST]
    r_wrist = landmarks[R_WRIST]
    l_elbow = landmarks[L_ELBOW]
    r_elbow = landmarks[R_ELBOW]
    l_hip = landmarks[L_HIP]
    r_hip = landmarks[R_HIP]

    wrist_low_left = l_wrist.y > l_elbow.y and abs(l_wrist.y - l_hip.y) < 0.2
    wrist_low_right = r_wrist.y > r_elbow.y and abs(r_wrist.y - r_hip.y) < 0.2

    if not (wrist_low_left and wrist_low_right):
        return False

    if prev_landmarks:

        dx1 = l_wrist.x - prev_landmarks[L_WRIST].x
        dy1 = l_wrist.y - prev_landmarks[L_WRIST].y
        dx2 = r_wrist.x - prev_landmarks[R_WRIST].x
        dy2 = r_wrist.y - prev_landmarks[R_WRIST].y

        dist = (dx1*dx1 + dy1*dy1) + (dx2*dx2 + dy2*dy2)

        if dist > 0.006:
            return False

    return True


def suggest_splits(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps

    skip_step = max(1, int(video_fps / TARGET_PROCESS_FPS))

    sys.stderr.write(
        f"INFO: {video_fps:.1f} FPS video | processing every {skip_step} frames\n")

    splits = []
    rest_start = None
    prev_landmarks = None

    frame_idx = 0
    last_progress = -1

    while True:

        # skip frames không decode
        for _ in range(skip_step - 1):
            if not cap.grab():
                break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += skip_step

        percent = int((frame_idx / frame_count) * 100)

        if percent // 5 != last_progress:
            sys.stderr.write(f"PROGRESS:{percent}\n")
            sys.stderr.flush()
            last_progress = percent // 5

        h, w = frame.shape[:2]

        # crop ngang giữ 1/2 trung tâm
        left = w // 4
        right = w * 3 // 4
        frame = frame[:, left:right]

        # crop dọc
        h, w = frame.shape[:2]
        top = int(h * 0.15)
        bottom = int(h * 0.9)
        frame = frame[top:bottom, :]

        # resize
        h, w = frame.shape[:2]
        new_h = int(h * RESIZE_WIDTH / w)
        frame = cv2.resize(frame, (RESIZE_WIDTH, new_h))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = pose.process(frame_rgb)

        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        current_time = frame_idx / video_fps

        is_rest = is_resting_pose(landmarks, prev_landmarks)

        if is_rest:

            if rest_start is None:
                rest_start = current_time

        else:

            if rest_start is not None:

                rest_duration = current_time - rest_start

                if rest_duration > 0.6:
                    split_time = rest_start + rest_duration / 2
                    splits.append(round(split_time, 3))

                rest_start = None

        prev_landmarks = landmarks

    if rest_start is not None and (duration - rest_start) > 0.5:
        splits.append(round(rest_start + (duration - rest_start) / 2, 3))

    cap.release()

    sys.stderr.write("PROGRESS:100\n")

    return {
        "splits": list(dict.fromkeys(splits)),
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