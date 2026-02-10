import pandas as pd
import numpy as np
import os
import pydicom
import cv2
import warnings

# ================= 1. 路徑設定 =================
RAW_TRAIN_PATH = '../../data/train.csv'
RAW_TEST_PATH = '../../data/test.csv'
PATH_PREFIX = '../../'

# ================= 2. 您的 ImplantDetector 類別 (完全保留) =================
class ImplantDetector:
    """
    金屬植入物檢測器
    利用金屬植入物在X光下高亮度的物理特性進行自動檢測和移除
    """

    def __init__(self, threshold=240, min_area=50, max_area=10000):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect_implants(self, image, return_stats=False):
        # 轉換為灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 基於亮度閾值的初步檢測
        high_intensity_mask = gray > self.threshold

        # 形態學處理：移除小噪聲
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(high_intensity_mask.astype(np.uint8),
                                       cv2.MORPH_OPEN, kernel)

        # 連通組件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8)

        final_mask = np.zeros_like(cleaned_mask, dtype=bool)
        valid_components = []

        for i in range(1, num_labels):  # 跳過背景
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                component_mask = (labels == i)
                final_mask |= component_mask
                valid_components.append({
                    'label': i,
                    'area': area
                })

        if return_stats:
            detection_stats = {
                'num_implants': len(valid_components),
                'image_coverage': np.sum(final_mask) / (image.shape[0] * image.shape[1]),
            }
            return final_mask, detection_stats

        return final_mask

# ================= 3. 影像讀取與前處理 =================
def load_dicom_for_detector(img_path):
    """
    讀取 DICOM 並確保其格式適合 ImplantDetector (類似 cv2.imread 的灰階效果)
    """
    if not os.path.exists(img_path):
        return None
    
    try:
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array.astype(float)
        
        # 為了配合 threshold=240 (0-255)，我們需要將 DICOM (通常是 12-bit 或 16-bit) 轉為 8-bit
        # 這裡採用 Robust Scaling: 將 99% 分位數視為 200，保留高光區給金屬
        # 這比 Min-Max 更能抵抗整張圖都很暗或都很亮的情況
        
        p99 = np.percentile(img, 99)
        if p99 == 0: p99 = img.max()
        
        # 映射邏輯：讓骨頭亮部 (p99) 落在 200 左右，金屬則會衝破 255 (被 clip)
        scale = 200.0 / p99
        img_8bit = img * scale
        img_8bit = np.clip(img_8bit, 0, 255).astype(np.uint8)
        
        return img_8bit
        
    except Exception as e:
        # print(f"Error reading {img_path}: {e}")
        return None

# ================= 4. 主程式 =================
def run_prevalence_analysis():
    print("=== Implant Prevalence Analysis (Using Original Logic) ===")
    
    # 使用您的預設參數
    detector = ImplantDetector(threshold=240, min_area=50, max_area=10000)
    
    for name, path in [("Training Set", RAW_TRAIN_PATH), ("Test Set", RAW_TEST_PATH)]:
        print(f"\nProcessing {name}...")
        try:
            df = pd.read_csv(path)
            implant_count = 0
            valid_images = 0
            
            for idx, row in df.iterrows():
                # 處理路徑
                rel_path = row['Img_path'].lstrip('/')
                full_path = os.path.join(PATH_PREFIX, rel_path)
                
                img = load_dicom_for_detector(full_path)
                
                if img is not None:
                    valid_images += 1
                    # 執行偵測
                    mask, stats = detector.detect_implants(img, return_stats=True)
                    
                    # 只要 num_implants > 0 就算有
                    if stats['num_implants'] > 0:
                        implant_count += 1
            
            if valid_images > 0:
                prev = implant_count / valid_images
                print(f"✅ {name}: {implant_count}/{valid_images} ({prev:.2%})")
            else:
                print(f"❌ No valid images found for {name}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_prevalence_analysis()