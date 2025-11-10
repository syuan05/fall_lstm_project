# 消融實驗組別：A2（Baseline, MAX_SEQ_LEN=144, y軸統一 0~1）
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# ==================== 基本參數設定 ====================
DATA_DIR = 'data'
LABEL_CSV = os.path.join(DATA_DIR, 'labels.csv')
MODEL_DIR = 'A3'
os.makedirs(MODEL_DIR, exist_ok=True)

# === 超參數設定 ===
MAX_SEQ_LEN = 200
FEATURE_DIM = 51
EPOCHS = 500
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
MODEL_NAME = 'A3'

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

# ==================== 切分資料集 ====================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
)
print(f"資料集比例：Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ==================== 模型架構 ====================
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, FEATURE_DIM)),

    LSTM(128, return_sequences=True),
    Dropout(DROPOUT_RATE),

    LSTM(64),

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
    monitor='val_loss', patience=30, restore_best_weights=True
)
csv_logger = CSVLogger(os.path.join(MODEL_DIR, f'{MODEL_NAME}_training_log.csv'))

# ==================== 訓練模型 ====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, csv_logger],
    verbose=1
)

# ==================== 儲存模型與訓練歷史 ====================
final_model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_final.keras')
model.save(final_model_path)
print(f"💾 最終模型已儲存至：{final_model_path}")

# 儲存訓練歷史（JSON + CSV）
history_path_json = os.path.join(MODEL_DIR, f'{MODEL_NAME}_history.json')
with open(history_path_json, 'w') as f:
    json.dump(history.history, f, indent=4)

history_path_csv = os.path.join(MODEL_DIR, f'{MODEL_NAME}_history.csv')
pd.DataFrame(history.history).to_csv(history_path_csv, index=False)
print(f"📊 訓練歷史已儲存：{history_path_json}, {history_path_csv}")

# ==================== 驗證與測試 ====================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\n✅ 驗證準確率：{val_acc:.4f} | 驗證損失：{val_loss:.4f}")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 測試準確率：{test_acc:.4f} | 測試損失：{test_loss:.4f}")

# 儲存 train/val/test 結果摘要
final_results_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_final_results.csv')
with open(final_results_path, 'w') as f:
    f.write('dataset,loss,accuracy\n')
    f.write(f'train,{history.history["loss"][-1]:.6f},{history.history["accuracy"][-1]:.6f}\n')
    f.write(f'val,{history.history["val_loss"][-1]:.6f},{history.history["val_accuracy"][-1]:.6f}\n')
    f.write(f'test,{test_loss:.6f},{test_acc:.6f}\n')
print(f"📄 最終結果已儲存：{final_results_path}")

# ==================== 混淆矩陣（Test） ====================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fall'])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title(f'Confusion Matrix - {MODEL_NAME}')
plt.savefig(os.path.join(MODEL_DIR, f'{MODEL_NAME}_confusion_matrix.png'))
plt.close()

# ==================== 分類報告（Test） ====================
report = classification_report(y_test, y_pred, target_names=['Normal', 'Fall'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_classification_report.csv')
report_df.to_csv(report_path, index=True)
print(f"📄 測試分類報告已儲存：{report_path}")

# ==================== 繪製訓練曲線（統一 y 軸 0~1） ====================
plt.figure(figsize=(12, 5))

# Loss 曲線
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'{MODEL_NAME} - Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.legend()

# Accuracy 曲線
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title(f'{MODEL_NAME} - Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_training_plot.png')
plt.savefig(plot_path)
plt.close()

# ==================== 輸出摘要 ====================
print("\n📊 測試結果摘要：")
print(report_df[['precision', 'recall', 'f1-score', 'support']])
print(f"\n📈 訓練曲線、分類報告、混淆矩陣、最終結果與歷史紀錄已儲存至：{MODEL_DIR}")
