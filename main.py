#  顔でマウスを操作するPythonシステム

import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipeの初期化（顔メッシュ）
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# カメラの初期化
cap = cv2.VideoCapture(0)

# 画面の幅と高さを取得（マウス移動用）
screen_width, screen_height = pyautogui.size()

# まばたき検出用変数
blink_timestamps = []  # まばたきのタイムスタンプを保存
blink_triggered = False  # 1回のまばたきで複数カウントしないように制御

# 鼻先移動の有効エリア（中央60%）
x_min, x_max = 0.2, 0.8
y_min, y_max = 0.2, 0.8

# 距離計算関数（2点間のユークリッド距離）
def get_distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # フレームを左右反転し、RGBに変換
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔メッシュの推定
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # 鼻先の座標（インデックス1）
        nose_tip = landmarks[1]
        x, y = nose_tip.x, nose_tip.y

        # 鼻先が中央60%の有効エリア内にある場合のみマウスを移動
        if x_min <= x <= x_max and y_min <= y <= y_max:
            relative_x = (x - x_min) / (x_max - x_min)
            relative_y = (y - y_min) / (y_max - y_min)
            screen_x = int(relative_x * screen_width)
            screen_y = int(relative_y * screen_height)
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        # まばたき検出（右目：上159、下145）
        eye_top = landmarks[159]
        eye_bottom = landmarks[145]
        eye_distance = get_distance(eye_top, eye_bottom)
        current_time = time.time()

        # まばたきが検出された場合（距離がしきい値より小さい）
        if eye_distance < 0.01:
            if not blink_triggered:
                blink_timestamps.append(current_time)
                blink_triggered = True
                # 直近2秒以内の記録のみ保持
                blink_timestamps = [t for t in blink_timestamps if current_time - t <= 2.0]
                print(f"まばたき回数（2秒以内）: {len(blink_timestamps)}")
        else:
            blink_triggered = False

        # 2回連続でまばたきされたら左クリック
        if len(blink_timestamps) >= 2:
            pyautogui.click()
            print("左クリック実行")
            blink_timestamps.clear()

        # 口を開けたかどうかの検出（上唇13, 下唇14）
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_distance = get_distance(mouth_top, mouth_bottom)

        # 距離がしきい値より大きければ右クリック
        if mouth_distance > 0.05:
            pyautogui.click(button='right')
            print("右クリック実行")
            time.sleep(0.5)  # 連続クリック防止

    # 有効エリアの表示（緑の枠で中央60%を可視化）
    h, w, _ = frame.shape
    cv2.rectangle(frame, (int(w * x_min), int(h * y_min)), (int(w * x_max), int(h * y_max)), (0, 255, 0), 2)
    cv2.putText(frame, "Area", (int(w * x_min), int(h * y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # ウィンドウにフレームを表示
    cv2.imshow("Air Mouse", frame)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
