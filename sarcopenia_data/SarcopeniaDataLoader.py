import sys
sys.path.extend(["../../", "../", "./"])
from einops import rearrange
from os.path import isdir
from torch.utils.data import Dataset
from commons.constant import *
from commons.utils import *
import os
import pandas as pd
import numpy as np
import torch
try:
    import pydicom
    from PIL import Image
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

# Import implant detector for metal implant removal
try:
    from .ImplantDetector import ImplantDetector
    IMPLANT_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from ImplantDetector import ImplantDetector
        IMPLANT_DETECTOR_AVAILABLE = True
    except ImportError:
        IMPLANT_DETECTOR_AVAILABLE = False
        print("Warning: ImplantDetector not available. Implant removal functionality disabled.")

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

# CSV-based dataset for regression tasks
class SarcopeniaCSVDataSet(Dataset):
    def __init__(self, csv_data, input_size=(224, 224), augment=True, text_only=False, contrastive_mode=False,
                 remove_implants=False, implant_threshold=240, removal_strategy='gaussian_noise',
                 shared_cache=None):  # Global image cache (optional)
        """
        CSV-based dataset for sarcopenia regression.

        Args:
            csv_data: pandas DataFrame with patient data
            input_size: Target image dimensions (height, width)
            augment: Whether to apply data augmentation
            text_only: If True, only load clinical features (no images)
            contrastive_mode: Enable contrastive learning (return two augmented views)
            remove_implants: Enable metal implant removal preprocessing
            implant_threshold: Intensity threshold for implant detection (200-250)
            removal_strategy: Method for implant removal ('gaussian_noise', 'zero', 'mean', etc.)
            shared_cache: GlobalImageCache instance for sharing images across folds
        """
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.augment = augment
        self.text_only = text_only
        self.contrastive_mode = contrastive_mode  # Enable contrastive learning mode
        self.data = csv_data.reset_index(drop=True)

        # Implant removal configuration
        self.remove_implants = remove_implants and IMPLANT_DETECTOR_AVAILABLE
        self.implant_threshold = implant_threshold
        self.removal_strategy = removal_strategy

        # Initialize implant detector if needed
        if self.remove_implants:
            self.implant_detector = ImplantDetector(threshold=implant_threshold)
            print(f"Implant removal enabled: threshold={implant_threshold}, strategy={removal_strategy}")
        else:
            self.implant_detector = None
            if remove_implants and not IMPLANT_DETECTOR_AVAILABLE:
                print("Warning: Implant removal requested but ImplantDetector not available.")

        # === Global image cache (shared across folds) ===
        self.shared_cache = shared_cache

        print(f'Load CSV samples: {len(self.data)}')
        if ASMI in self.data.columns:
            asmi_values = self.data[ASMI].values
            print(f'ASMI range: {asmi_values.min():.2f} - {asmi_values.max():.2f}')

    def __len__(self):
        return len(self.data)

    def load_dicom_image(self, img_path):
        try:
            # Read DICOM file
            dicom = pydicom.dcmread(img_path)
            img_array = dicom.pixel_array.astype(np.float32)

            # Normalize to 0-255 range
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
            img_array = img_array.astype(np.uint8)

            # === Implant Removal (before any other processing) ===
            if self.remove_implants and self.implant_detector is not None:
                # Work on grayscale version for implant detection
                if len(img_array.shape) == 3:
                    gray_for_detection = img_array[:,:,0]  # Use first channel
                else:
                    gray_for_detection = img_array

                # Detect and remove implants
                cleaned_array, implant_mask = self.implant_detector.remove_implants(
                    gray_for_detection, strategy=self.removal_strategy
                )

                # Apply cleaning to all channels if color image
                if len(img_array.shape) == 3:
                    for c in range(img_array.shape[2]):
                        img_array[:,:,c] = cleaned_array
                else:
                    img_array = cleaned_array

            # Convert to PIL Image for transforms
            # 如果維度是 2 (只有高度和寬度)，就代表這是一張灰階影像。
            if len(img_array.shape) == 2:
                # Grayscale to RGB
                img_array = np.stack([img_array, img_array, img_array], axis=2)

            # Resize to target size
            img = Image.fromarray(img_array.astype(np.uint8))
            img = img.resize((self.input_x, self.input_y))
            img_array = np.array(img)

            # Convert to torch tensor format (C, H, W)
            # 歸一化。將像素值從 [0, 255] 的範圍，縮放到 [0.0, 1.0] 的範圍，有助於穩定訓練過程。
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0
            return img_tensor

        except Exception as e:
            print(f"Warning: Error loading DICOM {img_path}: {e}")
            # Return dummy image on error
            return torch.zeros((3, self.input_y, self.input_x))

    def __getitem__(self, index):
        row = self.data.iloc[index] # 這個被取出來的 row，它的資料型別是 pandas Series。可以想成key:value的字典

        # Load image
        if not self.text_only and PATH in row and pd.notna(row[PATH]):
            original_path = row[PATH]  # Keep original path for cache lookup
            img_path = original_path

            # Adjust path for relative execution (e.g., from driver/reg_driver/)
            if not os.path.isabs(img_path) and not os.path.exists(img_path):
                # Try with ../../ prefix for execution from driver/reg_driver/
                potential_path = os.path.join("../../", img_path)
                if os.path.exists(potential_path):
                    img_path = potential_path

            # === Use global cache if available, otherwise load from disk ===
            if self.shared_cache is not None:
                # Use ORIGINAL path as key (cache stores with original paths)
                cached = self.shared_cache.get(original_path)
                if cached is not None:
                    # Load from global cache (fast! shared across all folds)
                    original_image = cached.clone()  # clone() to avoid in-place modifications
                else:
                    # Fallback: load from disk if not in cache (rare edge case)
                    original_image = self.load_dicom_image(img_path)
            else:
                # No cache: load from disk every time (slower)
                original_image = self.load_dicom_image(img_path)

            # Apply augmentation if specified
            if self.augment:
                # Convert from (C, H, W) to (H, W, C) numpy for augmentation
                img_np = (original_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # For contrastive learning, create two different augmented versions
                if self.contrastive_mode:
                    # Apply first augmentation
                    image_patch_1 = self.augment(img_np.copy())
                    # Apply second augmentation
                    image_patch_2 = self.augment(img_np.copy())

                    # Ensure both are float tensors and normalized
                    if not isinstance(image_patch_1, torch.Tensor):
                        image_patch_1 = torch.from_numpy(image_patch_1.transpose(2, 0, 1)).float() / 255.0
                    if not isinstance(image_patch_2, torch.Tensor):
                        image_patch_2 = torch.from_numpy(image_patch_2.transpose(2, 0, 1)).float() / 255.0
                else:
                    # Normal single augmentation
                    image_patch = self.augment(img_np)
                    if not isinstance(image_patch, torch.Tensor):
                        image_patch = torch.from_numpy(image_patch.transpose(2, 0, 1)).float() / 255.0
            else:
                # No augmentation
                if self.contrastive_mode:
                    # Use same image for both patches when no augmentation
                    image_patch_1 = original_image
                    image_patch_2 = original_image
                else:
                    image_patch = original_image
        else:
            # Dummy image for text-only mode or missing path
            if self.contrastive_mode:
                image_patch_1 = torch.zeros((3, self.input_y, self.input_x))
                image_patch_2 = torch.zeros((3, self.input_y, self.input_x))
            else:
                image_patch = torch.zeros((3, self.input_y, self.input_x))
        
        # Extract clinical features (text)
        clinical_features = []
        for col in TEXT_COLS: 
            if col in row and pd.notna(row[col]): # 這裡判斷了col是否有在row這個可想成字典的key中
                clinical_features.append(float(row[col]))
            else:
                clinical_features.append(0.0)  # Default value for missing features
        
        clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32)
        clinical_tensor = rearrange(clinical_tensor, 'b -> () () b')  # Match PTH format
        
        # Prepare return data based on mode
        if self.contrastive_mode:
            result = {
                "image_patch_1": image_patch_1,
                "image_patch_2": image_patch_2,
                "image_no": str(row[NO]) if NO in row else str(index),
                "image_uid": str(row[UID]) if UID in row else str(index),
                "image_path": row[PATH] if PATH in row else "unknown",
                "image_text": clinical_tensor,
            }
        else:
            result = {
                "image_patch": image_patch,
                "image_no": str(row[NO]) if NO in row else str(index),
                "image_uid": str(row[UID]) if UID in row else str(index),
                "image_path": row[PATH] if PATH in row else "unknown",
                "image_text": clinical_tensor,
            }

        # Add regression target (ASMI)
        if ASMI in row:
            result["image_asmi"] = torch.tensor(float(row[ASMI]), dtype=torch.float32)

        return result

def load_csv_data(data_path, train_filename, test_filename):
    """
    Load pre-split train and test CSV files
    
    Args:
        data_path: Path to the data directory
        train_filename: Training CSV filename
        test_filename: Test CSV filename
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    """
    import os
    
    # Construct full paths
    train_path = os.path.join(data_path, train_filename)
    test_path = os.path.join(data_path, test_filename)
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")
    
    # Load training data
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.columns = [col.strip() for col in train_df.columns]  # Clean column names
    
    # Load test data
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.columns = [col.strip() for col in test_df.columns]  # Clean column names
    
    
    print(f"CSV data loaded from separate files:")
    print(f"  - Training data: {len(train_df)} samples from {train_filename}")
    # print(f"  - Test data: {len(test_df)} samples from {test_filename}")
    # print(f"  - Total: {len(train_df) + len(test_df)} samples")
    
    return train_df, test_df

if __name__ == '__main__':
    print()