import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import ModelCheckpoint

# === 參數設定 ===
DATA_DIR = 'data'  # 資料夾路徑（內含 .npy 和 labels.csv）
LABEL_CSV = os.path.join(DATA_DIR, 'labels.csv')
MAX_SEQ_LEN = 160
FEATURE_DIM = 34  # 若有 confidence 則改為 51

# === 讀取 labels.csv ===
df = pd.read_csv(LABEL_CSV)

X_list, y_list = [], []

for _, row in df.iterrows():
    filename = row['filename'] + '.npy'
    label = row['label']
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        arr = np.load(path)
        X_list.append(arr)
        y_list.append(label)
    else:
        print(f"⚠️ 找不到檔案: {path}")

# === 補齊時間長度
X = pad_sequences(X_list, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post')
y = np.array(y_list)

# === 切分訓練與驗證資料
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 建立 LSTM 模型
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, FEATURE_DIM)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 建立 models 資料夾
os.makedirs('models_200rounds', exist_ok=True)

# === 模型儲存 Checkpoint
checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# === 訓練模型
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=00,
    batch_size=32,
    callbacks=[checkpoint]
)

# === 儲存最終模型
model.save('models/fall_lstm_model_final.h5')

# === 評估
loss, acc = model.evaluate(X_val, y_val)
print(f"✅ 評估準確率：{acc:.4f}")

# === 畫訓練圖
plt.figure(figsize=(12, 5))

# Loss 圖
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 圖
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_plot.png')
plt.close()
