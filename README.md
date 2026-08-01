# Gesture-Based Touchless Human-Computer Interaction

Hệ thống nhận diện cử chỉ tay real-time để điều khiển máy tính không tiếp xúc, sử dụng **MediaPipe** để trích xuất đặc trưng và **SVM** để phân loại cử chỉ.

Hỗ trợ 2 mode: **PowerPoint Presentation Controller** và **SolidWorks View Controller**.

---

## Ý nghĩa

Touchless HCI có ứng dụng trong:
- **Y tế** — bác sĩ thao tác máy tính trong phòng mổ mà không mất vô trùng
- **Công nghiệp** — kỹ sư điều khiển phần mềm kỹ thuật (CAD, FEA) mà không cần tháo găng tay bảo hộ
- **Giáo dục** — giảng viên điều khiển slide từ xa khi đứng bảng
- **Hỗ trợ người khuyết tật** — người bị hạn chế vận động tay vẫn tương tác được với máy tính

---

## Pipeline

```
Webcam
  ↓
MediaPipe Hands
  → 21 landmarks × (x, y, z) = 63 features
  → Normalize theo wrist (landmark 0)
  ↓
SVM Classifier (RBF kernel)
  → GridSearch tối ưu C, gamma
  → Confidence threshold = 0.70
  ↓
MediaPipe Verify
  → Đếm số ngón tay thẳng
  → Exact match với SVM prediction
  ↓
Gesture → Action (PPT hoặc SolidWorks)
```

---

## Gesture Set

### PPT Mode
| Gesture | Số ngón | Action |
|---------|---------|--------|
| `point` | 1 ngón trỏ | Next slide (→) |
| `2finger` | 2 ngón | Prev slide (←) |
| `3finger` | 3 ngón | Bắt đầu slideshow (F5) |
| `4finger` | 4 ngón | Kết thúc slideshow (Esc) |
| `open` | 5 ngón | Tạm dừng nhận diện |
| `fist` | 0 ngón | Tiếp tục nhận diện |

### SolidWorks Mode
| Gesture | Số ngón | Action |
|---------|---------|--------|
| `point` | 1 ngón | Front view (Ctrl+1) |
| `2finger` | 2 ngón | Back view (Ctrl+2) |
| `3finger` | 3 ngón | Left view (Ctrl+3) |
| `4finger` | 4 ngón | Right view (Ctrl+4) |
| `open` | 5 ngón | Top view (Ctrl+5) |
| `fist` | 0 ngón | Isometric view (Ctrl+7) |

---

## Cài đặt

**Yêu cầu:** Python 3.11

```bash
pip install -r requirements.txt
```

---

## Cấu trúc project

```
gesture_project/
│
├── app.py                  # Ứng dụng chính (PPT + SolidWorks)
├── collect_data.py         # Script thu thập data
├── enroll.py               # Đăng ký tay sử dụng
├── gesture_svm.ipynb       # Notebook train SVM
├── svm_gesture.pkl         # Model đã train
├── gesture_data.csv        # Dataset landmark
├── requirements.txt
└── README.md
```

---

## Hướng dẫn sử dụng

### 1. Thu thập data

```bash
python collect_data.py
```

| Phím | Chức năng |
|------|-----------|
| `1`–`6` | Chọn gesture cần thu |
| `Space` | Bắt đầu / dừng ghi |
| `R` | Xem thống kê |
| `Q` | Thoát |

Mục tiêu: **300 mẫu/gesture**, thu trong nhiều điều kiện ánh sáng và góc tay khác nhau.

### 2. Train SVM

Mở `gesture_svm.ipynb` trên Google Colab, upload `gesture_data.csv` và chạy toàn bộ notebook.

Download `svm_gesture.pkl` về máy sau khi train xong.

### 3. Chạy ứng dụng

Đặt `svm_gesture.pkl` cùng thư mục với `app.py`, sau đó:

```bash
python app.py
```

| Phím | Chức năng |
|------|-----------|
| `S` | Bật/tắt skeleton |
| `Q` | Thoát |

Dùng nút **P** / **S** trên màn hình để switch mode.

---

## Dataset

- **6 người** thu thập (đa dạng giới tính, kích thước bàn tay)
- **6 gesture class**, ~300 mẫu/gesture/người
- **~10,800 mẫu** tổng cộng sau khi ghép

Ghép nhiều file CSV:

```python
import pandas as pd, glob

files = glob.glob("*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("gesture_data_combined.csv", index=False)
```

---

## Kết quả

| Metric | Giá trị |
|--------|---------|
| Test Accuracy | **98.74%** |
| Model | SVM, kernel RBF |
| Features | 63 (21 landmarks × x,y,z) |
| Classes | 6 |
| FPS (realtime) | ~80 FPS |

---

## Toán học

| Thành phần | Công thức |
|-----------|-----------|
| Normalize landmark | $f_i = (x_i - x_0,\ y_i - y_0,\ z_i - z_0)$ |
| SVM kernel RBF | $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ |
| Cursor smoothing (EMA) | $\hat{x}_t = \alpha x_t + (1-\alpha)\hat{x}_{t-1}$ |
| Spread (zoom detect) | $s = \frac{1}{5}\sum_{i \in tips}\sqrt{(x_i-c_x)^2+(y_i-c_y)^2}$ |

---

## Hướng phát triển

- Tích hợp **SolidWorks API** (win32com) để điều khiển viewport trực tiếp thay vì hotkey
- Mở rộng gesture set với cử chỉ động (LSTM/HMM)
- **Calibration session** — thu thêm data riêng cho từng người dùng để fine-tune model
- Deploy trên **Raspberry Pi** cho ứng dụng nhúng

---