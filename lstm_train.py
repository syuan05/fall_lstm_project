import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# === 路徑設定 ===
X_PATH = 'data/X.npy'
y_PATH = 'data/y.npy'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_fall_model.h5')
PLOT_PATH = os.path.join(MODEL_DIR, 'training_plot.png')
EPOCHS = 70
BATCH_SIZE = 32

# === 確保模型資料夾存在 ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === 載入資料 ===
X = np.load(X_PATH)
y = np.load(y_PATH)

print("✅ Loaded data:", X.shape, y.shape)

# === 分類標籤處理 ===
y = to_categorical(y, num_classes=2)  # 二分類

# === 切分訓練與驗證集 ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 模型建立 ===
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 分類

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 模型訓練 ===
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

# === 繪製訓練曲線 ===
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print(f"✅ Model saved to {MODEL_PATH}")
print(f"📈 Training plot saved to {PLOT_PATH}")
