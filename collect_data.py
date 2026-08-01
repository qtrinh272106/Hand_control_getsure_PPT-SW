"""
collect_data.py — Thu thập landmark data cho SVM gesture recognition
=====================================================================
Gesture set:
  point   → 1 ngón trỏ
  2finger → 2 ngón (trỏ + giữa)
  3finger → 3 ngón
  4finger → 4 ngón (drag)
  open    → 5 ngón
  fist    → idle

Cách dùng:
  python collect_data.py

Phím tắt trong cửa sổ:
  1-6     → chọn gesture cần thu
  SPACE   → bắt đầu / dừng ghi
  Q       → thoát
  R       → xem thống kê hiện tại
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from collections import Counter

# ── Cấu hình ────────────────────────────────────────────────
OUTPUT_CSV   = "gesture_data.csv"
SAMPLES_PER_GESTURE = 300   # mục tiêu mỗi class
TIP_INDICES  = [4, 8, 12, 16, 20]  # đầu 5 ngón tay

GESTURES = {
    "1": "point",
    "2": "2finger",
    "3": "3finger",
    "4": "4finger",
    "5": "open",
    "6": "fist",
}

COLORS = {
    "point"  : (255, 200,   0),
    "2finger": (  0, 255,   0),
    "3finger": (  0, 200, 255),
    "4finger": (255,   0, 200),
    "open"   : (  0, 255, 200),
    "fist"   : (  0,   0, 255),
}

# ── MediaPipe setup ──────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    """
    Trích xuất 63 features từ 21 landmark.
    Normalize: lấy wrist (index 0) làm gốc tọa độ.
    
    Returns: list 63 float [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """
    lm     = hand_landmarks.landmark
    wrist  = lm[0]
    feats  = []
    for point in lm:
        feats.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z,
        ])
    return feats


def compute_centroid(hand_landmarks):
    """
    Tính centroid của 5 đầu ngón tay.
    Dùng cho scroll detection (toán thuần túy).
    
    Returns: (cx, cy) normalized [0,1]
    """
    lm   = hand_landmarks.landmark
    tips = [lm[i] for i in TIP_INDICES]
    cx   = sum(t.x for t in tips) / 5
    cy   = sum(t.y for t in tips) / 5
    return cx, cy


def load_existing_counts():
    """Đọc CSV hiện có để biết đã có bao nhiêu mẫu mỗi class."""
    if not os.path.exists(OUTPUT_CSV):
        return Counter()
    counts = Counter()
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                counts[row[-1]] += 1
    return counts


def draw_ui(frame, current_gesture, recording, counts, fps):
    h, w = frame.shape[:2]

    # ── Background panel ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Title ────────────────────────────────────────────────
    cv2.putText(frame, "GESTURE COLLECTOR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Gesture list ─────────────────────────────────────────
    y = 65
    for key, gesture in GESTURES.items():
        count   = counts.get(gesture, 0)
        pct     = min(count / SAMPLES_PER_GESTURE, 1.0)
        is_cur  = gesture == current_gesture
        color   = COLORS[gesture]

        # Highlight current
        if is_cur:
            cv2.rectangle(frame, (5, y - 16), (275, y + 6), color, -1)
            txt_color = (0, 0, 0)
        else:
            txt_color = color

        label = f"[{key}] {gesture:<8} {count:>3}/{SAMPLES_PER_GESTURE}"
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

        # Progress bar
        bar_x = 10
        bar_y = y + 9
        cv2.rectangle(frame, (bar_x, bar_y), (270, bar_y + 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(260 * pct), bar_y + 4), color, -1)
        y += 40

    # ── Status ───────────────────────────────────────────────
    cv2.line(frame, (5, y), (275, y), (80, 80, 80), 1)
    y += 20

    if recording:
        # Blinking dot
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (20, y), 7, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING  [SPACE]=stop", (35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
    else:
        cv2.putText(frame, "[SPACE] start recording", (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    y += 25
    cv2.putText(frame, f"FPS: {fps:.0f}  [Q] quit  [R] stats", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # ── Current gesture big label (top right) ────────────────
    if current_gesture:
        color = COLORS[current_gesture]
        cv2.putText(frame, current_gesture.upper(),
                    (w - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return frame


def print_stats(counts):
    print("\n─── Thống kê hiện tại ───────────────────────────────")
    total = 0
    for gesture in GESTURES.values():
        n   = counts.get(gesture, 0)
        pct = n / SAMPLES_PER_GESTURE * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {gesture:<10} {bar} {n:>4}/{SAMPLES_PER_GESTURE} ({pct:.0f}%)")
        total += n
    print(f"\n  Tổng cộng: {total} mẫu")
    print("─────────────────────────────────────────────────────\n")


def main():
    # ── Khởi tạo CSV ─────────────────────────────────────────
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="")
    writer      = csv.writer(csv_file)

    if not file_exists:
        # Header: x0,y0,z0,...,x20,y20,z20, label
        header = [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
        header.append("label")
        writer.writerow(header)
        print(f"✅ Tạo file mới: {OUTPUT_CSV}")
    else:
        print(f"✅ Append vào file: {OUTPUT_CSV}")

    counts          = load_existing_counts()
    current_gesture = "point"
    recording       = False
    saved_count     = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps       = 0

    print("\n─── Hướng dẫn ───────────────────────────────────────")
    print("  Phím 1-6 : chọn gesture")
    print("  SPACE    : bắt đầu / dừng ghi")
    print("  R        : xem thống kê")
    print("  Q        : thoát")
    print("─────────────────────────────────────────────────────")
    print_stats(counts)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── FPS ──────────────────────────────────────────
            now      = time.time()
            fps      = 0.9 * fps + 0.1 * (1 / max(now - prev_time, 1e-9))
            prev_time = now

            # ── MediaPipe ────────────────────────────────────
            result   = hands.process(rgb)
            detected = False

            if result.multi_hand_landmarks:
                hand_lm  = result.multi_hand_landmarks[0]
                detected = True

                # Vẽ skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1),
                )

                # Vẽ centroid
                cx, cy = compute_centroid(hand_lm)
                h, w   = frame.shape[:2]
                cv2.circle(frame, (int(cx * w), int(cy * h)), 8,
                           (0, 255, 0), -1)
                cv2.putText(frame,
                            f"centroid ({cx:.2f}, {cy:.2f})",
                            (int(cx * w) + 12, int(cy * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0), 1)

                # ── Ghi data ─────────────────────────────────
                if recording and current_gesture:
                    feats = extract_features(hand_lm)
                    writer.writerow(feats + [current_gesture])
                    csv_file.flush()
                    counts[current_gesture] += 1
                    saved_count += 1

                    # Thông báo mỗi 50 mẫu
                    if saved_count % 50 == 0:
                        print(f"  [{current_gesture}] {counts[current_gesture]} mẫu đã lưu")

                    # Tự dừng khi đủ mẫu
                    if counts[current_gesture] >= SAMPLES_PER_GESTURE:
                        recording = False
                        print(f"\n✅ [{current_gesture}] Đủ {SAMPLES_PER_GESTURE} mẫu!")
                        print_stats(counts)

            else:
                # Không detect tay → không ghi
                if recording:
                    cv2.putText(frame, "⚠ Không thấy tay!", (300, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

            # ── UI ───────────────────────────────────────────
            frame = draw_ui(frame, current_gesture, recording and detected,
                            counts, fps)

            cv2.imshow("Gesture Data Collector", frame)

            # ── Keyboard ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif chr(key) in GESTURES:
                current_gesture = GESTURES[chr(key)]
                recording       = False
                saved_count     = 0
                print(f"\n→ Chuyển sang gesture: {current_gesture}")

            elif key == ord(" "):
                if not detected:
                    print("⚠ Chưa detect được tay, đưa tay vào frame trước!")
                else:
                    recording   = not recording
                    saved_count = 0
                    if recording:
                        print(f"\n🔴 Bắt đầu ghi [{current_gesture}]...")
                    else:
                        print(f"⏹ Dừng ghi [{current_gesture}]")
                        print_stats(counts)

            elif key == ord("r"):
                print_stats(counts)

    # ── Cleanup ──────────────────────────────────────────────
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print("\n─── Hoàn tất ────────────────────────────────────────")
    print_stats(counts)
    print(f"  File đã lưu: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
