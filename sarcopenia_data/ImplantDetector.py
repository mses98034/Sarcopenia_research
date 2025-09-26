"""
植入物檢測和移除工具
基於亮度閾值的金屬植入物自動檢測，用於消除模型對植入物的依賴偏差
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, filters
import warnings

class ImplantDetector:
    """
    金屬植入物檢測器

    利用金屬植入物在X光下高亮度的物理特性進行自動檢測和移除
    """

    def __init__(self, threshold=240, min_area=50, max_area=10000):
        """
        初始化植入物檢測器

        Args:
            threshold (int): 亮度閾值，預設240 (0-255範圍)
            min_area (int): 最小植入物區域像素數，過濾噪聲
            max_area (int): 最大植入物區域像素數，避免誤刪大片區域
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect_implants(self, image, return_stats=False):
        """
        檢測影像中的金屬植入物區域

        Args:
            image (numpy.ndarray): 輸入影像 (H, W) 或 (H, W, C)
            return_stats (bool): 是否返回檢測統計信息

        Returns:
            mask (numpy.ndarray): 植入物遮罩 (H, W)，植入物區域為True
            stats (dict, optional): 檢測統計信息
        """
        # 轉換為灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 基於亮度閾值的初步檢測
        high_intensity_mask = gray > self.threshold

        # 形態學處理：移除小噪聲，保持植入物主體
        # 使用開運算移除小的高亮點
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(high_intensity_mask.astype(np.uint8),
                                       cv2.MORPH_OPEN, kernel)

        # 連通組件分析：濾除不符合大小範圍的區域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8)

        final_mask = np.zeros_like(cleaned_mask, dtype=bool)
        valid_components = []

        for i in range(1, num_labels):  # 跳過背景(label=0)
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                component_mask = (labels == i)
                final_mask |= component_mask
                valid_components.append({
                    'label': i,
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                })

        if return_stats:
            detection_stats = {
                'num_implants': len(valid_components),
                'total_implant_pixels': np.sum(final_mask),
                'image_coverage': np.sum(final_mask) / (image.shape[0] * image.shape[1]),
                'components': valid_components,
                'threshold_used': self.threshold
            }
            return final_mask, detection_stats

        return final_mask

    def create_mask(self, image, morphology_filter=True, return_stats=False):
        """
        創建植入物遮罩（detect_implants的別名，保持向後兼容）
        """
        return self.detect_implants(image, return_stats=return_stats)

    def remove_implants(self, image, strategy='gaussian_noise', mask=None):
        """
        從影像中移除植入物

        Args:
            image (numpy.ndarray): 輸入影像
            strategy (str): 移除策略
                - 'zero': 設為0（黑化）
                - 'mean': 用影像平均值填充
                - 'gaussian_noise': 用高斯噪聲填充
                - 'inpaint': 使用OpenCV修復算法
                - 'median_background': 用周圍區域中位數填充
            mask (numpy.ndarray, optional): 預計算的植入物遮罩

        Returns:
            cleaned_image (numpy.ndarray): 清理後的影像
            mask (numpy.ndarray): 使用的植入物遮罩
        """
        if mask is None:
            mask = self.detect_implants(image)

        cleaned_image = image.copy()

        if not np.any(mask):
            # 無植入物檢測到，返回原影像
            return cleaned_image, mask

        if strategy == 'zero':
            cleaned_image[mask] = 0

        elif strategy == 'mean':
            # 使用非植入物區域的平均值
            background_mean = np.mean(image[~mask])
            cleaned_image[mask] = background_mean

        elif strategy == 'gaussian_noise':
            # 使用背景區域的統計特性生成噪聲
            background_pixels = image[~mask]
            bg_mean = np.mean(background_pixels)
            bg_std = np.std(background_pixels)

            # 生成與植入物區域形狀相同的高斯噪聲
            noise_shape = np.sum(mask)
            noise = np.random.normal(bg_mean, bg_std, noise_shape)

            # 確保噪聲在合理範圍內
            if len(image.shape) == 3:
                noise = np.clip(noise, 0, 255)
            else:
                noise = np.clip(noise, 0, 255)

            cleaned_image[mask] = noise

        elif strategy == 'inpaint':
            # 使用OpenCV的修復算法
            if len(image.shape) == 3:
                # 彩色影像：分別對每個通道進行修復
                for c in range(image.shape[2]):
                    cleaned_image[:,:,c] = cv2.inpaint(
                        image[:,:,c].astype(np.uint8),
                        mask.astype(np.uint8),
                        inpaintRadius=3,
                        flags=cv2.INPAINT_TELEA
                    )
            else:
                # 灰階影像
                cleaned_image = cv2.inpaint(
                    image.astype(np.uint8),
                    mask.astype(np.uint8),
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA
                )

        elif strategy == 'median_background':
            # 用植入物周圍區域的中位數填充
            # 擴展遮罩以獲取周圍區域
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
            surrounding_region = expanded_mask.astype(bool) & (~mask)

            if np.any(surrounding_region):
                median_val = np.median(image[surrounding_region])
                cleaned_image[mask] = median_val
            else:
                # 無周圍區域，回退到平均值策略
                background_mean = np.mean(image[~mask])
                cleaned_image[mask] = background_mean

        else:
            raise ValueError(f"Unknown removal strategy: {strategy}")

        return cleaned_image, mask

    def visualize_detection(self, image, mask=None, save_path=None):
        """
        可視化植入物檢測結果

        Args:
            image (numpy.ndarray): 原始影像
            mask (numpy.ndarray, optional): 植入物遮罩
            save_path (str, optional): 儲存路徑

        Returns:
            visualization (numpy.ndarray): 可視化影像
        """
        if mask is None:
            mask = self.detect_implants(image)

        # 轉換為RGB格式以便可視化
        if len(image.shape) == 3:
            vis_image = image.copy()
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 在植入物區域疊加紅色標記
        vis_image[mask] = [255, 0, 0]  # 紅色標記植入物

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        return vis_image

    def get_implant_statistics(self, image):
        """
        獲取植入物檢測的詳細統計信息

        Args:
            image (numpy.ndarray): 輸入影像

        Returns:
            stats (dict): 詳細統計信息
        """
        mask, stats = self.detect_implants(image, return_stats=True)

        # 添加額外的統計信息
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if np.any(mask):
            implant_intensities = gray[mask]
            stats.update({
                'implant_intensity_mean': np.mean(implant_intensities),
                'implant_intensity_std': np.std(implant_intensities),
                'implant_intensity_max': np.max(implant_intensities),
                'implant_intensity_min': np.min(implant_intensities)
            })
        else:
            stats.update({
                'implant_intensity_mean': 0,
                'implant_intensity_std': 0,
                'implant_intensity_max': 0,
                'implant_intensity_min': 0
            })

        return stats

    @staticmethod
    def compare_removal_strategies(image, detector, strategies=['zero', 'mean', 'gaussian_noise', 'inpaint']):
        """
        比較不同植入物移除策略的效果

        Args:
            image (numpy.ndarray): 輸入影像
            detector (ImplantDetector): 檢測器實例
            strategies (list): 要比較的策略列表

        Returns:
            results (dict): 各策略的結果影像
        """
        mask = detector.detect_implants(image)
        results = {'original': image, 'mask': mask}

        for strategy in strategies:
            try:
                cleaned_image, _ = detector.remove_implants(image, strategy=strategy, mask=mask)
                results[strategy] = cleaned_image
            except Exception as e:
                warnings.warn(f"Strategy {strategy} failed: {e}")
                results[strategy] = None

        return results