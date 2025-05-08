import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(1)
screen_width, screen_height = pyautogui.size()
safe_margin = 5
pyautogui.FAILSAFE = True

# 鼻先座標の動的キャリブレーション範囲
nose_x_min, nose_x_max = 1.0, 0.0
nose_y_min, nose_y_max = 1.0, 0.0

# 状態管理用変数
blink_timestamps = []
blink_triggered = False
last_right_click_time = 0

# 2点間の距離計算
def get_distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

print("鼻先を上下左右に動かして、画面全体の制御キャリブレーションを行ってください。")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        nose = landmarks[1]

        # 顔のランドマーク描画
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
        )

        # 鼻先の座標を使ってマウス位置をキャリブレーション
        nose_x_min = min(nose_x_min, nose.x)
        nose_x_max = max(nose_x_max, nose.x)
        nose_y_min = min(nose_y_min, nose.y)
        nose_y_max = max(nose_y_max, nose.y)

        if (nose_x_max - nose_x_min) > 0.01 and (nose_y_max - nose_y_min) > 0.01:
            relative_x = (nose.x - nose_x_min) / (nose_x_max - nose_x_min)
            relative_y = (nose.y - nose_y_min) / (nose_y_max - nose_y_min)

            screen_x = int(relative_x * screen_width)
            screen_y = int(relative_y * screen_height)

            screen_x = max(safe_margin, min(screen_width - safe_margin, screen_x))
            screen_y = max(safe_margin, min(screen_height - safe_margin, screen_y))

            pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        # まばたき検出（右目：159と145）
        eye_top = landmarks[159]
        eye_bottom = landmarks[145]
        eye_distance = get_distance(eye_top, eye_bottom)
        current_time = time.time()

        if eye_distance < 0.01:
            if not blink_triggered:
                blink_timestamps.append(current_time)
                blink_triggered = True
                blink_timestamps = [t for t in blink_timestamps if current_time - t <= 2.0]
        else:
            blink_triggered = False

        if len(blink_timestamps) >= 2:
            pyautogui.click()
            print("左クリックが実行されました。")
            blink_timestamps.clear()

        # 口の開閉検出（13と14）
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_distance = get_distance(mouth_top, mouth_bottom)

        if mouth_distance > 0.05 and current_time - last_right_click_time > 1.0:
            pyautogui.click(button='right')
            print("右クリックが実行されました。")
            last_right_click_time = current_time

        # デバッグ用テキスト表示
        cv2.putText(frame, f"X: {nose.x:.3f} [{nose_x_min:.2f}, {nose_x_max:.2f}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Y: {nose.y:.3f} [{nose_y_min:.2f}, {nose_y_max:.2f}]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ウィンドウ表示
    cv2.imshow("AirMouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
