# ASMI 預測網頁應用程式

基於 Gradio 的網頁應用程式，使用深度學習從髖部 X 光影像和臨床參數預測四肢骨骼肌指數（ASMI）。

## 🎯 功能特色

- **多模態預測**：結合髖部 X 光影像與臨床特徵（年齡、性別、身高、體重、BMI）
- **肌少症風險評估**：使用性別特定的 AWGS 閾值自動分類
- **模型可解釋性**：Grad-CAM 視覺化顯示 X 光影像中影響預測的區域
- **友善使用介面**：簡潔直觀的 Gradio 網頁 UI
- **多種圖片格式**：支援 DICOM (.dcm)、PNG 和 JPEG 檔案

## 📋 需求

### 系統需求
- Python 3.8 或更高版本
- 建議 4GB+ RAM
- GPU 非必要（支援 CPU 推理）

### Python 相依套件

完整清單請參閱 `requirements.txt`。主要相依套件：
- torch >= 2.0.0
- torchvision >= 0.15.0
- gradio >= 4.0.0
- pydicom >= 2.3.0
- opencv-python >= 4.8.0

## 🚀 快速開始

### 1. 安裝

```bash
# 進入 app 目錄
cd /path/to/MM-CL/app/

# 安裝相依套件
pip install -r requirements.txt
```

### 2. 執行應用程式

```bash
# 啟動 Gradio 網頁應用
python app.py
```

應用程式將會：
1. 載入訓練好的模型（首次執行可能需要 10-30 秒）
2. 啟動本地網頁伺服器
3. 自動在瀏覽器開啟 `http://localhost:7860`

### 3. 使用應用程式

**步驟 1：上傳影像**
- 點擊影像上傳區域
- 選擇髖部 X 光影像（**建議使用 DICOM (.dcm) 格式以獲得最準確的預測**）
- 也支援 PNG 或 JPG 格式（但可能因壓縮損失導致預測精度下降）

**步驟 2：輸入臨床資訊**
- 年齡（歲）：18-120
- 性別：男性或女性
- 身高（公分）：100-250
- 體重（公斤）：30-300
- BMI：選填（留空時自動計算）

**步驟 3：預測**
- 點擊「🔮 Predict ASMI」按鈕
- 等待數秒處理
- 查看結果和 CAM 視覺化

## 📊 理解輸出結果

### 1. 預測 ASMI
- 連續數值，單位為 kg/m²
- 代表四肢骨骼肌質量除以身高平方

### 2. 肌少症風險評估
- **Yes**：ASMI 低於性別特定閾值（表示有風險）
- **No**：ASMI 高於閾值（正常範圍）

**參考閾值（AWGS 標準）：**
- 男性：< 7.0 kg/m²
- 女性：< 5.4 kg/m²

### 3. CAM 視覺化
- 熱圖疊加在 X 光影像上
- **紅色區域**：模型最關注的區域
- **藍色區域**：影響較小的區域
- 協助解釋哪些解剖特徵驅動了預測

## 🔧 設定

### 模型路徑
應用程式使用最佳訓練模型，位於：
```
../log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/checkpoint/best_model.pth
```

若要使用不同模型，請編輯 `config.py`：
```python
BEST_MODEL_PATH = "/path/to/your/model.pth"
```

### 裝置選擇
應用程式會自動偵測最佳可用裝置（CUDA > MPS > CPU）。

強制使用 CPU，請編輯 `config.py`：
```python
DEVICE = torch.device('cpu')
```

### 肌少症閾值
修改診斷閾值，請編輯 `config.py`：
```python
SARCOPENIA_THRESHOLDS = {
    "Male": 7.0,
    "Female": 5.4
}
```

## 🌐 線上部署

### 選項 1：Hugging Face Spaces（免費）

1. 在 [huggingface.co](https://huggingface.co) 建立帳號
2. 建立新的 Space（Gradio SDK）
3. 上傳 `app/` 目錄下的所有檔案
4. 使用 Git LFS 上傳模型檔案（`best_model.pth`）
5. Space 自動建置和部署

**重要**：針對 Spaces 部署更新 `config.py` 中的 `BEST_MODEL_PATH`：
```python
BEST_MODEL_PATH = "best_model.pth"  # Spaces 使用相對路徑
```

### 選項 2：本地網路分享

啟用公開 URL 進行臨時分享：

在 `app.py` 中修改：
```python
demo.launch(
    share=True,  # 建立臨時公開 URL
    ...
)
```

### 選項 3：Docker 部署

建立 `Dockerfile`：
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

建置和執行：
```bash
docker build -t asmi-app .
docker run -p 7860:7860 asmi-app
```

## 🧪 測試

### 測試模型載入
```bash
python model_loader.py
```

### 測試前處理
```python
from preprocessing import preprocess_image, calculate_bmi

# 測試影像載入
img_tensor = preprocess_image("path/to/test_image.png")
print(f"影像張量形狀：{img_tensor.shape}")

# 測試 BMI 計算
bmi = calculate_bmi(height_cm=170, weight_kg=70)
print(f"BMI: {bmi}")
```

### 測試推理
```python
from inference import predict_asmi

asmi, risk, cam = predict_asmi(
    "path/to/image.png",
    age=65,
    gender="Male",
    height=170,
    weight=70,
    bmi=24.2
)
print(f"ASMI: {asmi:.2f} kg/m²")
print(f"肌少症風險：{risk}")
```

## ⚠️ 限制與免責聲明

### 臨床使用
- **本工具僅供研究目的使用**
- 未經臨床診斷核准
- 不應取代專業醫療評估
- 預測結果應由合格醫療專業人員解讀

### 技術限制
- 模型訓練於特定族群（可能無法泛化）
- 影像品質影響預測準確度
- CAM 視覺化顯示相關性而非因果關係
- 單一視角 X 光可能遺漏重要資訊

### 輸入需求
- 髖部 X 光影像（建議前後位）
- 清晰顯示髖部區域
- 足夠的影像品質（避免過度雜訊/偽影）

## 📚 模型資訊

### 架構
- **模型**：ResNetFusionAttentionNetRegression
- **骨幹網路**：ResNet-34（在 ImageNet 上預訓練）
- **融合機制**：門控注意力機制
- **輸入**：224×224 灰階影像 + 5 個臨床特徵
- **輸出**：連續 ASMI 數值（kg/m²）

### 訓練細節
- **損失函數**：一致性相關係數（CCC）
- **優化器**：AdamW
- **交叉驗證**：5-fold
- **資料增強**：旋轉、平移、水平翻轉

### 效能指標
詳細的內部和外部驗證效能指標請參閱主專案 README 或發表論文。

## 🐛 疑難排解

### 常見問題

**1. 找不到模型檔案**
```
FileNotFoundError: Model file not found: ...
```
**解決方案**：確認 `config.py` 中的 `BEST_MODEL_PATH` 指向正確的模型檔案。

**2. CUDA 記憶體不足**
```
RuntimeError: CUDA out of memory
```
**解決方案**：在 `config.py` 中強制使用 CPU 或減少批次大小。

**3. DICOM 載入失敗**
```
RuntimeError: pydicom is not installed
```
**解決方案**：安裝 pydicom：`pip install pydicom`

**4. Import 錯誤**
```
ModuleNotFoundError: No module named 'models'
```
**解決方案**：確保從 `app/` 目錄執行，且父專案結構完整。

## 📧 支援

針對此網頁應用程式的問題：
1. 查看此 README 的常見解決方案
2. 確認所有相依套件已安裝
3. 確保模型檔案存在且可存取

針對底層模型或研究問題，請參閱主 MM-CL 專案文件。

## 📄 授權

本應用程式承襲主 MM-CL 專案的授權。詳情請參閱專案的 LICENSE 檔案。

## 🙏 致謝

- 深度學習模型：MM-CL（Multi-modality Contrastive Learning）框架
- 網頁框架：Gradio by Hugging Face
- 視覺化：TorchCAM for Grad-CAM 實作

---

**版本**：1.0.0
**最後更新**：2025-10-08
