import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# --- 1. Đọc dữ liệu ---
data = pd.read_csv("Du_lieu_da_loc.csv")  # ← thay đường dẫn nếu cần
data = data.dropna()

# --- 2. Tách input/output ---
X = data.iloc[:, 1:-1].values  # Bỏ cột Time và cột cuối
y = data.iloc[:, -1].values    # Cột cuối là công suất P_solar

# --- 3. Chuẩn hóa dữ liệu ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# --- 4. Lưu mean/std và scaler_y để dùng khi dự đoán ---
joblib.dump(scaler_X.mean_, "scaler_X_mean.pkl")
joblib.dump(scaler_X.scale_, "scaler_X_std.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("✅ scaler_X_mean, scaler_X_std và scaler_y.pkl đã được lưu.")

# --- 5. Tách tập train/val ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# --- 6. Xây dựng mô hình ANN ---
model = Sequential([
    Input(shape=(X_train.shape[1],)),        # 👈 KHÔNG dùng input_shape trong Dense
    Dense(300, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # đầu ra: P_solar
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- 7. Cài EarlyStopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# --- 8. Huấn luyện mô hình ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# --- 9. Lưu mô hình ---
model.save("ann_model.h5")
print("✅ Mô hình đã được lưu vào ann_model.h5")

# --- 10. Vẽ biểu đồ loss ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Biểu đồ Loss trong quá trình huấn luyện ANN")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
