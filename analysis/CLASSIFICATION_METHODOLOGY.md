# Classification Analysis Methodology

## 📊 正確的肌少症分類分析方法

### 核心概念

我們的模型是**回歸模型**（預測ASMI值），但需要評估其**診斷肌少症的能力**（二分類任務）。

---

## 🎯 三個關鍵組成部分

### 1. Ground Truth (y_true)

**來源**: CSV檔案中的 `Low_muscle_mass` 欄位

```python
y_true = df['Low_muscle_mass'].values.astype(int)
# 1 = 肌少症, 0 = 正常
```

**為什麼**:
- ✅ 這是醫學專家標註的診斷結果
- ✅ 已考慮所有臨床因素（不只是ASMI）
- ✅ 是真正的ground truth

**❌ 錯誤做法**:
- 不要從 `actual_asmi` 重新計算（可能因小數點誤差不一致）

---

### 2. Predicted Binary Labels (y_pred)

**計算方式**: 從預測的ASMI值 + 性別特異性閾值

```python
if Gender == 0 (Male):
    y_pred = 1 if predicted_asmi < 7.0 else 0

if Gender == 1 (Female):
    y_pred = 1 if predicted_asmi < 5.5 else 0
```

**用途**:
- Confusion Matrix
- Accuracy
- Sensitivity (Recall)
- Specificity
- F1 Score
- PPV (Precision)
- NPV

---

### 3. ROC Score (y_score)

**計算方式**: 標準化距離分數

```python
def calculate_risk_score(predicted_asmi, gender):
    threshold = 7.0 if gender == 0 else 5.5
    # 距離閾值的距離
    return threshold - predicted_asmi

y_score = [calculate_risk_score(asmi, gender)
           for asmi, gender in zip(predicted_asmi, Gender)]
```

**解釋**:
- **Positive score** (>0): ASMI低於閾值 → 肌少症風險 ✅
- **Negative score** (<0): ASMI高於閾值 → 正常 ✅
- **Score絕對值**: 距離閾值的遠近

**用途**:
- ROC Curve
- AUC-ROC

**為什麼這樣做**:
- ✅ ROC曲線需要**連續的score**
- ✅ 考慮性別差異（男女閾值不同）
- ✅ 統計上直接、可解釋
- ✅ 沒有人為的sigmoid轉換

**ROC圖上的Threshold Operating Point**:

在ROC曲線上，我們用**紅色星號(★)**標記使用性別特異性閾值時的operating point：

```python
# 計算使用threshold時的TPR和FPR
tn = sum((y_true == 0) & (y_pred == 0))
fp = sum((y_true == 0) & (y_pred == 1))
fn = sum((y_true == 1) & (y_pred == 0))
tp = sum((y_true == 1) & (y_pred == 1))

threshold_fpr = fp / (fp + tn)  # False Positive Rate at threshold
threshold_tpr = tp / (tp + fn)  # True Positive Rate at threshold

# 在ROC曲線上標記這個點
plt.scatter(threshold_fpr, threshold_tpr, marker='*', s=150, color='red')
```

這個點顯示：
- 當我們使用Male <7.0, Female <5.5作為診斷標準時
- 模型在ROC空間中的performance位置
- 這是實際臨床應用時的operating point

---

## 📈 完整程式碼範例

```python
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# 1. 從CSV讀取ground truth
y_true = df['Low_muscle_mass'].values.astype(int)

# 2. 計算predicted binary labels
predicted_asmi = df['predicted_asmi'].values
genders = df['Gender'].values

y_pred = np.zeros(len(predicted_asmi), dtype=int)
male_mask = (genders == 0)
female_mask = (genders == 1)
y_pred[male_mask] = (predicted_asmi[male_mask] < 7.0).astype(int)
y_pred[female_mask] = (predicted_asmi[female_mask] < 5.5).astype(int)

# 3. 計算ROC score
def calculate_risk_score(asmi, gender):
    threshold = 7.0 if gender == 0 else 5.5
    return threshold - asmi

y_score = np.array([
    calculate_risk_score(asmi, gender)
    for asmi, gender in zip(predicted_asmi, genders)
])

# 4. 計算指標
# Binary metrics
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# ROC metrics
auc = roc_auc_score(y_true, y_score)
```

---

## 📝 論文Methods部分建議寫法

```
Sarcopenia Classification Analysis

To evaluate the model's diagnostic performance for sarcopenia, we
performed binary classification analysis using the predicted ASMI
values and gender-specific diagnostic thresholds based on the Asian
Working Group for Sarcopenia (AWGS) criteria (male: ASMI < 7.0 kg/m²,
female: ASMI < 5.5 kg/m²).

Ground truth labels were obtained from clinical diagnoses recorded
in the dataset (Low_muscle_mass column). Predicted binary labels
were determined by applying the gender-specific ASMI thresholds to
the model's predicted ASMI values.

For ROC curve analysis, we calculated a continuous risk score as
the standardized distance from the gender-specific threshold:

    Risk Score = Threshold - Predicted ASMI

where Threshold = 7.0 kg/m² for males and 5.5 kg/m² for females.
This approach accounts for gender-specific diagnostic criteria while
maintaining the continuous nature required for ROC analysis. Positive
scores indicate ASMI values below the threshold (sarcopenia), while
negative scores indicate values above the threshold (normal).

We calculated sensitivity, specificity, accuracy, F1-score, positive
predictive value (PPV), and negative predictive value (NPV) using
the binary predictions. The area under the ROC curve (AUC-ROC) was
computed using the continuous risk scores.
```

---

## ✅ 驗證檢查清單

執行分析前請確認：

1. **CSV檔案包含必要欄位**:
   - ✅ `Low_muscle_mass`: Ground truth
   - ✅ `predicted_asmi`: 模型預測值
   - ✅ `Gender`: 性別 (0=Male, 1=Female)

2. **數據一致性**:
   - ✅ Ground truth使用CSV的Low_muscle_mass（不重新計算）
   - ✅ 預測標籤使用predicted_asmi（不是actual_asmi）

3. **ROC分析**:
   - ✅ 使用連續的risk score（不是binary predictions）
   - ✅ Score考慮性別差異
   - ✅ Score方向正確（positive = sarcopenia risk）

---

## 🔬 常見錯誤

### ❌ 錯誤1: 從actual_asmi重新計算ground truth
```python
# 錯誤！
y_true = (actual_asmi < gender_threshold).astype(int)
```

**為什麼錯**:
- 可能與CSV的Low_muscle_mass不一致
- 醫學診斷不只看ASMI一個指標

### ❌ 錯誤2: 使用binary predictions做ROC
```python
# 錯誤！ROC需要連續score
y_score = y_pred  # 這只會得到一個點，不是曲線
```

### ❌ 錯誤3: 不考慮性別差異
```python
# 錯誤！男女閾值不同
y_score = -predicted_asmi  # 沒有標準化到同一尺度
```

---

## 📊 預期結果

- **Training set (Cross-validation)**: AUC ~0.7-0.8
- **Test set (External)**: AUC ~0.7-0.8
- **Confusion matrix**: 使用真實ground truth vs predicted binary
- **ROC curve**: 使用連續risk score

---

## 📚 參考文獻

1. Asian Working Group for Sarcopenia (AWGS) 2019 Consensus
2. Fawcett, T. (2006). "An introduction to ROC analysis". Pattern Recognition Letters.
3. DeLong, E.R., et al. (1988). "Comparing the areas under two or more correlated ROC curves"
