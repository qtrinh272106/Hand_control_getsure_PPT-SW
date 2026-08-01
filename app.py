"""
app.py — Gesture Controller (PPT + SolidWorks)
===============================================
Pipeline: Webcam → MediaPipe (63 landmark) → SVM → gesture → action

Mode PPT (M để switch):
  point   (1 ngon) → next slide  (→)
  2finger (2 ngon) → prev slide  (←)
  3finger (3 ngon) → bat dau trinh chieu (F5)
  4finger (4 ngon) → ket thuc trinh chieu (Esc)
  open    (5 ngon) → tam dung nhan dien
  fist    (0 ngon) → tiep tuc nhan dien

Mode SolidWorks (click nut de switch):
  point   (1 ngon) → Front view  (Ctrl+1)
  2finger (2 ngon) → Back view   (Ctrl+2)
  3finger (3 ngon) → Left view   (Ctrl+3)
  4finger (4 ngon) → Right view  (Ctrl+4)
  open    (5 ngon) → Top view    (Ctrl+5)
  fist    (0 ngon) → Isometric   (Ctrl+7)

Pham tat:
  M → doi mode (PPT / SolidWorks)
  S → bat/tat skeleton
  Q → thoat
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import time
import threading
import tkinter as tk
import os
from collections import deque

# ── Cau hinh ────────────────────────────────────────────────
MODEL_PATH      = "svm_gesture.pkl"
CONF_THRESHOLD  = 0.70
GESTURE_BUFFER  = 5
GESTURE_LOCK    = 2
HOLD_DURATION   = 0.3

CAM_W, CAM_H    = 640, 480
PROCESS_W       = 320
TIP_INDICES     = [4, 8, 12, 16, 20]

# ── Load model ───────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded!")

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

lock   = threading.Lock()
shared = {"hand_lm": None, "frame": None, "running": True}

gesture_buffer   = deque(maxlen=GESTURE_BUFFER)
current_gesture  = None
change_streak    = 0
change_candidate = None

hold = {
    "start_time"  : None,
    "fired"       : False,
    "prev_gesture": None,
}




# Global panel — khởi tạo trong main()
panel = None
class ControlPanel:
    """
    Cửa sổ nhỏ góc trên phải:
    - Toast notification (tự ẩn sau 1.5s)
    - Button switch mode PPT / SolidWorks
    """
    def __init__(self, on_mode_switch):
        self._root        = None
        self._toast_label = None
        self._mode_btn    = None
        self._after_id    = None
        self._on_switch   = on_mode_switch
        t = threading.Thread(target=self._run_tk, daemon=True)
        t.start()
        time.sleep(0.15)

    def _run_tk(self):
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.configure(bg="#1a1a1a")
        self._root.resizable(False, False)

        # Toast label
        self._toast_label = tk.Label(
            self._root, text="",
            font=("Segoe UI", 11, "bold"),
            
            fg="white", bg="#1a1a1a",
            padx=12, pady=3,
        )
        self._toast_label.pack(fill="x")

        # Divider
        tk.Frame(self._root, bg="#333333", height=1).pack(fill="x")

        # P / S toggle buttons
        btn_frame = tk.Frame(self._root, bg="#1a1a1a", pady=4)
        btn_frame.pack()

        self._btn_p = tk.Button(
            btn_frame, text="P",
            font=("Segoe UI", 10, "bold"),
            fg="white", bg="#0078d4",
            activebackground="#005a9e",
            activeforeground="white",
            relief="flat", width=3, pady=4,
            cursor="hand2",
            command=lambda: self._set_mode("PPT"),
        )
        self._btn_p.pack(side="left", padx=(6, 2))

        self._btn_s = tk.Button(
            btn_frame, text="S",
            font=("Segoe UI", 10, "bold"),
            fg="#888888", bg="#2a2a2a",
            activebackground="#3a3a3a",
            activeforeground="white",
            relief="flat", width=3, pady=4,
            cursor="hand2",
            command=lambda: self._set_mode("SW"),
        )
        self._btn_s.pack(side="left", padx=(2, 6))

        self._reposition()
        self._root.mainloop()

    def _set_mode(self, mode):
        if self._on_switch:
            self._root.after(0, lambda: self._on_switch_mode(mode))

    def _on_switch_mode(self, mode):
        # Callback với mode cụ thể
        self._on_switch(mode)

    def _reposition(self):
        if not self._root:
            return
        self._root.update_idletasks()
        sw = self._root.winfo_screenwidth()
        w  = max(self._root.winfo_reqwidth(), 190)
        h  = self._root.winfo_reqheight()
        self._root.geometry(f"{w}x{h}+{sw - w - 20}+20")

    def show_toast(self, message):
        if self._root:
            self._root.after(0, lambda: self._show(message))

    def _show(self, message):
        if self._after_id:
            self._root.after_cancel(self._after_id)
        self._toast_label.config(text=message)
        self._reposition()
        self._after_id = self._root.after(1500, self._hide_toast)

    def _hide_toast(self):
        if self._toast_label:
            self._toast_label.config(text="")
        self._reposition()
        self._after_id = None

    def update_mode(self, mode):
        if self._root:
            self._root.after(0, lambda: self._update_btns(mode))

    def _update_btns(self, mode):
        if mode == "PPT":
            self._btn_p.config(fg="white",   bg="#0078d4")
            self._btn_s.config(fg="#888888", bg="#2a2a2a")
        else:
            self._btn_p.config(fg="#888888", bg="#2a2a2a")
            self._btn_s.config(fg="white",   bg="#d45a00")
        self._reposition()

    def destroy(self):
        if self._root:
            self._root.after(0, self._root.destroy)


# ════════════════════════════════════════════════════════════
# CAMERA THREAD
# ════════════════════════════════════════════════════════════
def camera_thread(cap, show_skeleton_ref):
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=0,
    ) as hands:
        while shared["running"]:
            ret, frame = cap.read()
            if not ret:
                continue
            frame  = cv2.flip(frame, 1)
            small  = cv2.resize(frame, (PROCESS_W, PROCESS_W * CAM_H // CAM_W))
            rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_lm = None
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                if show_skeleton_ref[0]:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 200, 255),
                                               thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(200, 200, 200),
                                               thickness=1),
                    )
            with lock:
                shared["hand_lm"] = hand_lm
                shared["frame"]   = frame


# ════════════════════════════════════════════════════════════
# GESTURE HELPERS
# ════════════════════════════════════════════════════════════
def extract_features(hand_landmarks):
    lm    = hand_landmarks.landmark
    wrist = lm[0]
    feats = []
    for point in lm:
        feats.extend([point.x - wrist.x,
                      point.y - wrist.y,
                      point.z - wrist.z])
    return feats


def compute_centroid(hand_landmarks):
    lm   = hand_landmarks.landmark
    tips = [lm[i] for i in TIP_INDICES]
    return sum(t.x for t in tips) / 5, sum(t.y for t in tips) / 5


def count_fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = [
        lm[8].y  < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y,
    ]
    thumb_up = lm[4].x < lm[3].x
    return sum(fingers) + (1 if thumb_up else 0)


def verify_gesture(svm_label, hand_landmarks):
    if svm_label is None:
        return None
    n = count_fingers_up(hand_landmarks)
    expected = {
        "open"   : 5, "4finger": 4, "3finger": 3,
        "2finger": 2, "point"  : 1, "fist"   : 0,
    }
    return svm_label if n == expected.get(svm_label, -1) else None


def predict_gesture(hand_landmarks):
    feats = extract_features(hand_landmarks)
    X     = np.array(feats).reshape(1, -1)
    label = model.predict(X)[0]
    conf  = model.predict_proba(X).max()
    if conf < CONF_THRESHOLD:
        return None, conf
    return verify_gesture(label, hand_landmarks), conf


def voted_gesture(new_label):
    global current_gesture, change_streak, change_candidate
    if new_label is not None:
        gesture_buffer.append(new_label)
    if not gesture_buffer:
        return None
    voted = max(set(gesture_buffer), key=gesture_buffer.count)
    if voted != current_gesture:
        if voted == change_candidate:
            change_streak += 1
        else:
            change_candidate = voted
            change_streak    = 1
        if change_streak >= GESTURE_LOCK:
            current_gesture  = voted
            change_streak    = 0
            change_candidate = None
    else:
        change_streak    = 0
        change_candidate = None
    return current_gesture


def update_hold(gesture):
    if gesture != hold["prev_gesture"]:
        hold["start_time"]   = time.time() if gesture else None
        hold["fired"]        = False
        hold["prev_gesture"] = gesture
    if gesture is None or hold["start_time"] is None:
        return 0.0, False
    elapsed     = time.time() - hold["start_time"]
    progress    = min(elapsed / HOLD_DURATION, 1.0)
    should_fire = (progress >= 1.0) and not hold["fired"]
    if should_fire:
        hold["fired"] = True
    return progress, should_fire


def reset_hold():
    hold["start_time"]   = None
    hold["fired"]        = False
    hold["prev_gesture"] = None


# ════════════════════════════════════════════════════════════
# ACTIONS
# ════════════════════════════════════════════════════════════
def do_ppt(gesture, paused):
    msg       = None
    pause_cmd = None

    if gesture == "open":
        if not paused:
            pause_cmd = "pause"
            msg       = "Tam dung nhan dien"
    elif gesture == "fist":
        if paused:
            pause_cmd = "resume"
            msg       = "Tiep tuc nhan dien"
    elif not paused:
        if gesture == "point":
            pyautogui.press("right")
            msg = "Next Slide  ->"
        elif gesture == "2finger":
            pyautogui.press("left")
            msg = "<-  Prev Slide"
        elif gesture == "3finger":
            pyautogui.press("f5")
            msg = "Bat dau trinh chieu"
        elif gesture == "4finger":
            pyautogui.press("escape")
            msg = "Ket thuc trinh chieu"

    if msg:
        panel.show_toast(msg)
    return msg, pause_cmd


def do_solidworks(gesture):
    """
    Dùng pyautogui simulate phím tắt SolidWorks.
    Trả về msg để hiện toast.
    """
    msg = None

    if gesture == "point":
        pyautogui.hotkey("ctrl", "1")
        msg = "Front View  (Ctrl+1)"
    elif gesture == "2finger":
        pyautogui.hotkey("ctrl", "2")
        msg = "Back View   (Ctrl+2)"
    elif gesture == "3finger":
        pyautogui.hotkey("ctrl", "3")
        msg = "Left View   (Ctrl+3)"
    elif gesture == "4finger":
        pyautogui.hotkey("ctrl", "4")
        msg = "Right View  (Ctrl+4)"
    elif gesture == "open":
        pyautogui.hotkey("ctrl", "5")
        msg = "Top View    (Ctrl+5)"
    elif gesture == "fist":
        pyautogui.hotkey("ctrl", "7")
        msg = "Isometric   (Ctrl+7)"

    if msg:
        panel.show_toast(msg)
    return msg


# ════════════════════════════════════════════════════════════
# DRAW UI
# ════════════════════════════════════════════════════════════
COLORS = {
    "open"   : (  0, 255, 200),
    "fist"   : (100, 100, 255),
    "2finger": (  0, 255,   0),
    "3finger": (  0, 200, 255),
    "4finger": (255, 200,   0),
    "point"  : (150, 180, 255),
}

MODE_LEGEND = {
    "PPT": [
        ("point",   "1 ngon  next slide ->"),
        ("2finger", "2 ngon  prev slide <-"),
        ("3finger", "3 ngon  bat dau  F5"),
        ("4finger", "4 ngon  ket thuc Esc"),
        ("open",    "5 ngon  tam dung"),
        ("fist",    "0 ngon  tiep tuc"),
    ],
    "SW": [
        ("point",   "1 ngon  Front  Ctrl+1"),
        ("2finger", "2 ngon  Back   Ctrl+2"),
        ("3finger", "3 ngon  Left   Ctrl+3"),
        ("4finger", "4 ngon  Right  Ctrl+4"),
        ("open",    "5 ngon  Top    Ctrl+5"),
        ("fist",    "0 ngon  Iso    Ctrl+7"),
    ],
}


def draw_ui(frame, gesture, conf, fps, paused, last_action,
            hold_progress, cam_ok, mode):
    h, w = frame.shape[:2]
    color = COLORS.get(gesture, (200, 200, 200)) if gesture else (70, 70, 70)

    # Panel trai
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (290, 230), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # Mode badge
    mode_color = (0, 200, 255) if mode == "PPT" else (255, 180, 0)
    cv2.rectangle(frame, (10, 6), (90, 26), mode_color, -1)
    cv2.putText(frame,
                "PPT MODE" if mode == "PPT" else "SW  MODE",
                (13, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(frame, "[M] doi mode",
                (98, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 130, 130), 1)

    # Trang thai pause (chi PPT)
    if mode == "PPT":
        st_col  = (0, 50, 230) if paused else (0, 200, 80)
        st_text = "PAUSED" if paused else "Dang nhan dien"
        cv2.putText(frame, st_text, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, st_col, 1)
    else:
        sw_col  = (0, 200, 80)
        sw_text = "SolidWorks (hotkey)"
        cv2.putText(frame, sw_text, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, sw_col, 1)


    # Gesture label
    cv2.putText(frame, gesture.upper() if gesture else "---",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    # Confidence bar
    y0 = 100
    if conf is not None:
        blen = int(conf * 160)
        cv2.rectangle(frame, (10, y0), (170, y0 + 10), (45, 45, 45), -1)
        cv2.rectangle(frame, (10, y0), (10 + blen, y0 + 10), color, -1)
        cv2.putText(frame, f"{conf:.0%}",
                    (175, y0 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # Hold progress bar
    y1 = y0 + 14
    if gesture and conf is not None:
        hlen   = int(hold_progress * 160)
        hcolor = (0, int(80 + hold_progress * 175), int((1 - hold_progress) * 220))
        cv2.rectangle(frame, (10, y1), (170, y1 + 8), (38, 38, 38), -1)
        if hlen > 0:
            cv2.rectangle(frame, (10, y1), (10 + hlen, y1 + 8), hcolor, -1)
        label = "FIRE!" if hold_progress >= 1.0 else f"{hold_progress:.0%}"
        cv2.putText(frame, label,
                    (175, y1 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, hcolor, 1)

    # FPS + last action
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1)
    if last_action:
        cv2.putText(frame, last_action,
                    (10, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 235, 120), 2)

    # Watermark PAUSED
    if paused and mode == "PPT":
        cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 210), 4)

    # Legend bên phải
    legend = MODE_LEGEND[mode]
    y_leg  = 18
    for g, desc in legend:
        c     = COLORS.get(g, (180, 180, 180))
        thick = 2 if g == gesture else 1
        cv2.putText(frame, desc,
                    (w - 272, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, thick)
        y_leg += 20

    # System status bar
    statuses = [
        ("CAMERA", "OK"     if cam_ok           else "ERROR",   (0, 200, 80) if cam_ok           else (0, 50, 220)),
        ("MODEL ", "OK",                                          (0, 200, 80)),
        ("STATUS", "ACTIVE" if not paused        else "PAUSED",  (0, 200, 80) if not paused       else (0, 50, 220)),
    ]
    px, py = w - 155, h - 68
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px - 8, py - 14), (w - 4, h - 4), (18, 18, 18), -1)
    cv2.addWeighted(ov2, 0.70, frame, 0.30, 0, frame)
    for i, (lbl, val, col) in enumerate(statuses):
        cv2.putText(frame, f"{lbl}: {val}",
                    (px, py + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

    cv2.putText(frame, "[Q] quit   [S] skeleton   [M] mode",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80, 80, 80), 1)
    return frame


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    global panel
    cap    = cv2.VideoCapture(0)
    cam_ok = cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    show_skeleton_ref = [True]
    t = threading.Thread(target=camera_thread,
                         args=(cap, show_skeleton_ref), daemon=True)
    t.start()

    mode          = ["PPT"]   # list để callback có thể sửa
    paused        = [False]
    prev_time     = time.time()
    fps           = 0.0
    last_conf     = None
    gesture       = None
    last_action   = None
    hold_prog     = 0.0
    prev_centroid = None

    def on_mode_switch(new_mode=None):
        if new_mode:
            mode[0] = new_mode
        else:
            mode[0] = "SW" if mode[0] == "PPT" else "PPT"
        gesture_buffer.clear()
        reset_hold()
        paused[0] = False
        panel.update_mode(mode[0])
        panel.show_toast(f"{'PowerPoint' if mode[0] == 'PPT' else 'SolidWorks'}")
        print(f"\n  >>> Doi sang mode: {mode[0]}\n")

    panel = ControlPanel(on_mode_switch)

    print("\n─── Gesture Controller ──────────────────────────────")
    print("  Click nut tren man hinh de doi mode PPT / SolidWorks")
    print("  [S] bat/tat skeleton")
    print("  [Q] thoat")
    print("─────────────────────────────────────────────────────\n")

    while True:
        with lock:
            frame   = shared["frame"]
            hand_lm = shared["hand_lm"]

        if frame is None:
            time.sleep(0.005)
            continue

        frame     = frame.copy()
        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-9)
        prev_time = now

        if hand_lm is not None:
            raw_label, conf = predict_gesture(hand_lm)
            last_conf       = conf
            gesture         = voted_gesture(raw_label)

            if mode[0] == "PPT":
                prev_centroid = None
                hold_prog, should_fire = update_hold(gesture)
                if gesture and should_fire:
                    msg, pause_cmd = do_ppt(gesture, paused[0])
                    if msg:
                        last_action = msg
                        panel.show_toast(msg)
                        print(f"  {msg}")
                    if pause_cmd == "pause":
                        paused[0] = True
                        gesture_buffer.clear()
                        reset_hold()
                    elif pause_cmd == "resume":
                        paused[0] = False
                        gesture_buffer.clear()
                        reset_hold()

            else:  # SW mode
                hold_prog = 0.0
                prev_centroid = None
                _, should_fire = update_hold(gesture)
                if gesture and should_fire:
                    msg = do_solidworks(gesture)
                    if msg:
                        last_action = msg
                        print(f"  {msg}")

        else:
            gesture_buffer.clear()
            gesture       = None
            last_conf     = None
            hold_prog     = 0.0
            prev_centroid = None
            reset_hold()

        frame = draw_ui(frame, gesture, last_conf, fps,
                        paused[0], last_action, hold_prog, cam_ok, mode[0])
        cv2.imshow("Gesture Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_skeleton_ref[0] = not show_skeleton_ref[0]

        if cv2.getWindowProperty("Gesture Controller",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    shared["running"] = False
    panel.destroy()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDa thoat!")


if __name__ == "__main__":
    main()