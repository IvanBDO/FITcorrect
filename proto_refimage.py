import cv2
import mediapipe as mp
import math
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ======================
# Image Paths
# ======================
pose_images = {
    "Body Squat Pose": cv2.imread("images/squat.png"),
    "Lunge Pose": cv2.imread("images/lunge.png"),
    "Push-Up Pose": cv2.imread("images/pushup.png"),
    "Sit-Up Pose": cv2.imread("images/situp.png"),
    "Tree Pose": cv2.imread("images/tree.png"),

    "Bodyweight Squat": cv2.imread("images/squat.png"),
    "Lunge": cv2.imread("images/lunge.png"),
    "Push-Up": cv2.imread("images/pushup.png"),
    "Sit-Up": cv2.imread("images/situp.png"),
    "Tree Pose": cv2.imread("images/tree.png"),
}

# ======================
# Overlay Helper
# ======================
def overlay_reference(frame, title, scale=0.3):
    if title not in pose_images or pose_images[title] is None:
        return frame
    ref_img = pose_images[title]
    h, w, _ = frame.shape
    ref_h, ref_w = int(h * scale), int(w * scale)
    ref_img = cv2.resize(ref_img, (ref_w, ref_h))
    x_offset, y_offset = w - ref_w - 10, h - ref_h - 10
    frame[y_offset:y_offset + ref_h, x_offset:x_offset + ref_w] = ref_img
    return frame

# ======================
# Helper Functions
# ======================
def calculate_angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        max(math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2), 1e-6))
    return math.degrees(math.acos(max(min(cos_angle, 1), -1)))

def calculate_torso_angle(shoulder, hip, knee):
    sh_hip = (shoulder.x - hip.x, shoulder.y - hip.y)
    hip_knee = (hip.x - knee.x, hip.y - knee.y)
    cos_angle = (sh_hip[0]*hip_knee[0] + sh_hip[1]*hip_knee[1]) / (
        max(math.sqrt(sh_hip[0]**2 + sh_hip[1]**2) * math.sqrt(hip_knee[0]**2 + hip_knee[1]**2), 1e-6))
    return math.degrees(math.acos(max(min(cos_angle,1),-1)))

def is_visible(lm, *indices, thresh=0.6):
    return all(lm[i].visibility > thresh for i in indices)

def is_standing(lm):
    shoulders_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                   lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    hips_y = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].y +
              lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    ankles_y = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y +
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2
    return shoulders_y < hips_y < ankles_y

# ======================
# Pose Check Functions
# ======================
def moderate_squat(lm):
    l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    hips_aligned = abs(l_hip.x - r_hip.x) < 0.05
    left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    arms_forward = left_arm_angle > 150 and right_arm_angle > 150
    return (left_knee_angle < 150 and right_knee_angle < 150 and hips_aligned and arms_forward)

def moderate_lunge(lm):
    if not is_visible(lm,
                      mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.RIGHT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value,
                      mp_pose.PoseLandmark.RIGHT_KNEE.value,
                      mp_pose.PoseLandmark.LEFT_ANKLE.value,
                      mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                      mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                      mp_pose.PoseLandmark.RIGHT_SHOULDER.value):
        return False
    l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    if l_knee.y < l_hip.y or r_knee.y < r_hip.y:
        return False
    hips_staggered = abs(l_hip.x - r_hip.x) > 0.05
    shoulders_level = abs(l_shoulder.y - r_shoulder.y) < 0.05
    return hips_staggered and shoulders_level

def lenient_pushup(lm):
    if is_standing(lm):
        return False
    l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    return (40 < left_elbow_angle < 150 and 40 < right_elbow_angle < 150)

def lenient_situp(lm):
    if is_standing(lm):
        return False
    l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_torso_angle = calculate_torso_angle(l_shoulder, l_hip, l_knee)
    right_torso_angle = calculate_torso_angle(r_shoulder, r_hip, r_knee)
    return (30 < left_torso_angle < 150 and 30 < right_torso_angle < 150)

def lenient_tree(lm):
    l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    return (left_arm_angle < 120 and right_arm_angle < 120 and abs(l_ankle.y - r_ankle.y) > 0.08)

# ======================
# Calibration
# ======================
def calibration_phase(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("⚠️ Camera not available.")
        return 1.0
    calibration_poses = [
        ("Body Squat Pose", moderate_squat),
        ("Lunge Pose", moderate_lunge),
        ("Push-Up Pose", lenient_pushup),
        ("Sit-Up Pose", lenient_situp),
        ("Tree Pose", lenient_tree)
    ]
    factor = 1.0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for title, check_func in calibration_poses:
            print(f"\n➡️ Show: {title}")
            holding = False
            hold_start = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.putText(image, f"Calibration: {title}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    if check_func(lm):
                        cv2.putText(image, "Good form!", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                        if not holding:
                            holding = True
                            hold_start = time.time()
                        hold_time = time.time()-hold_start
                        cv2.putText(image, f"Holding {hold_time:.1f}s", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                        if hold_time >= 2:
                            time.sleep(5)
                            break
                    else:
                        holding = False
                        cv2.putText(image, "Adjust form", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                else:
                    holding = False
                    cv2.putText(image, "No body detected", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

                image = overlay_reference(image, title)
                cv2.imshow("Calibration", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                elif key == ord('n'):
                    time.sleep(5)
                    break
    cap.release()
    cv2.destroyWindow("Calibration")
    return factor

# ======================
# Exercise Loop
# ======================
def exercise_loop(camera_index=0, factor=1.0):
    print("\n=== Exercise Flow Started ===")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"⚠️ Camera {camera_index} not available.")
        return
    exercises = [
        ("Bodyweight Squat", moderate_squat),
        ("Lunge", moderate_lunge),
        ("Push-Up", lenient_pushup),
        ("Sit-Up", lenient_situp),
        ("Tree Pose", lenient_tree)
    ]
    current_ex = 0
    rep_state = {ex: "up" for ex, _ in exercises}
    rep_counts = {ex: 0 for ex, _ in exercises}
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame,1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            title, ex_func = exercises[current_ex]
            cv2.putText(image, f"Exercise: {title}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if ex_func(landmarks):
                    cv2.putText(image, "Good form!", (50,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                else:
                    cv2.putText(image, "Adjust form", (50,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                image = overlay_reference(image, title)
                if title == "Bodyweight Squat":
                    angle = calculate_angle(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                    if angle < 110 and rep_state[title]=="up":
                        rep_state[title]="down"
                    elif angle > 150 and rep_state[title]=="down":
                        rep_state[title]="up"; rep_counts[title]+=1
                elif title == "Lunge":
                    angle = calculate_angle(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                    if angle < 110 and rep_state[title]=="up":
                        rep_state[title]="down"
                    elif angle > 150 and rep_state[title]=="down":
                        rep_state[title]="up"; rep_counts[title]+=1
                elif title == "Push-Up":
                    angle = calculate_angle(
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                    if angle < 80 and rep_state[title]=="up":
                        rep_state[title]="down"
                    elif angle > 150 and rep_state[title]=="down":
                        rep_state[title]="up"; rep_counts[title]+=1
                elif title == "Sit-Up":
                    angle = calculate_torso_angle(
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
                    if angle < 80 and rep_state[title]=="up":
                        rep_state[title]="down"
                    elif angle > 130 and rep_state[title]=="down":
                        rep_state[title]="up"; rep_counts[title]+=1
                if title != "Tree Pose":
                    cv2.putText(image, f"Reps: {rep_counts[title]}", (50,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            else:
                cv2.putText(image, "Full body not detected", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Exercise Correction', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_ex = (current_ex + 1) % len(exercises)
                time.sleep(5)
    cap.release()
    cv2.destroyAllWindows()

# ======================
# Main
# ======================
if __name__ == "__main__":
    LENIENCY_FACTOR = calibration_phase()
    exercise_loop(camera_index=0, factor=LENIENCY_FACTOR)