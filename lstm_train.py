
# 消融實驗組別：A2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==================== 基本參數設定 ====================
DATA_DIR = 'data'
LABEL_CSV = os.path.join(DATA_DIR, 'labels.csv')
MODEL_DIR = 'models_baseline'
os.makedirs(MODEL_DIR, exist_ok=True)

# === 超參數設定（可作為後續消融基準）===
MAX_SEQ_LEN = 160
FEATURE_DIM = 51          # 若不含 confidence 改為 34
EPOCHS = 300
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
MODEL_NAME = 'baseline'

# ==================== 讀取資料 ====================
df = pd.read_csv(LABEL_CSV)
X_list, y_list = [], []

for _, row in df.iterrows():
    path = os.path.join(DATA_DIR, row['filename'] + '.npy')
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[1] != FEATURE_DIM:
            arr = arr[:, :FEATURE_DIM]
        X_list.append(arr)
        y_list.append(row['label'])
    else:
        print(f"⚠️ 找不到檔案: {path}")

X = pad_sequences(X_list, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post')
y = np.array(y_list)

# === 切分訓練 / 驗證集 ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==================== 模型架構 ====================
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, FEATURE_DIM)),
    LSTM(64, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==================== Callback ====================
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, f'{MODEL_NAME}_best.keras'),
    save_best_only=True, monitor='val_loss', mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# ==================== 訓練模型 ====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

final_model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_final.keras')
model.save(final_model_path)
print(f"💾 最終模型已儲存至：{final_model_path}")

# ==================== 評估與輸出 ====================
loss, acc = model.evaluate(X_val, y_val)
print(f"\n✅ 驗證準確率：{acc:.4f} | 驗證損失：{loss:.4f}")

# === 混淆矩陣 ===
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

cm = confusion_matrix(y_val, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fall'])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title(f'Confusion Matrix - {MODEL_NAME}')
plt.savefig(os.path.join(MODEL_DIR, f'{MODEL_NAME}_confusion_matrix.png'))
plt.close()

# === 分類報告 ===
report = classification_report(y_val, y_pred, target_names=['Normal', 'Fall'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_classification_report.csv')
report_df.to_csv(report_path, index=True)
print(f"📄 分類報告已儲存：{report_path}")

# === 繪製訓練曲線 ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'{MODEL_NAME} - Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title(f'{MODEL_NAME} - Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_training_plot.png')
plt.savefig(plot_path)
plt.close()

# === 輸出摘要結果 ===
print("\n📊 訓練結果摘要：")
print(report_df[['precision', 'recall', 'f1-score', 'support']])
print(f"\n📈 訓練曲線與混淆矩陣已儲存至：{MODEL_DIR}")

