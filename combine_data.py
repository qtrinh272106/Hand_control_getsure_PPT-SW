import pandas as pd
import glob
import os

# Tìm tất cả CSV trong thư mục hiện tại
files = glob.glob("*.csv")
print(f"Tìm thấy {len(files)} file: {files}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Tổng mẫu: {len(df)}")
print(df["label"].value_counts())

df.to_csv("gesture_data_combined.csv", index=False)
print("✅ Đã lưu: gesture_data_combined.csv")