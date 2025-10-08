"""
Image preprocessing module for ASMI prediction
Handles DICOM and common image formats
"""
import numpy as np
import torch
from PIL import Image
import cv2
import sys
import os

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM support disabled.")

from config import IMAGE_SIZE, PROJECT_ROOT

# Add project root to path for importing ImplantDetector
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import ImplantDetector
try:
    from sarcopenia_data.ImplantDetector import ImplantDetector
    IMPLANT_DETECTOR_AVAILABLE = True
    # Initialize global detector (threshold=240, same as training)
    _global_implant_detector = ImplantDetector(threshold=240)
except ImportError:
    IMPLANT_DETECTOR_AVAILABLE = False
    _global_implant_detector = None
    print("Warning: ImplantDetector not available. Implant removal disabled.")


def load_dicom_image(dicom_path, remove_implants=True):
    """
    Load DICOM image (matches training preprocessing in SarcopeniaDataLoader.py)

    Args:
        dicom_path: Path to DICOM file
        remove_implants: Whether to remove implants (default True, matches training)

    Returns:
        numpy array: Image normalized to 0-255 range (H, W), uint8
    """
    if not DICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed. Cannot load DICOM files.")

    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize to 0-255 range (same as training)
    # Do NOT apply DICOM windowing - not used during training
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)

    # Apply implant removal if enabled (same as training)
    if remove_implants and IMPLANT_DETECTOR_AVAILABLE and _global_implant_detector is not None:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray_for_detection = img[:, :, 0]
        else:
            gray_for_detection = img

        # Remove implants (strategy='inpaint' matches training config)
        cleaned_img, _ = _global_implant_detector.remove_implants(
            gray_for_detection,
            strategy='inpaint'
        )

        # Apply cleaning to all channels if color image
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                img[:, :, c] = cleaned_img
        else:
            img = cleaned_img

    return img


def load_image_file(image_path):
    """
    Load common image formats (PNG, JPG, etc.)

    Args:
        image_path: Path to image file

    Returns:
        numpy array: Grayscale image (H, W)
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = np.array(img).astype(np.float32)

    # Normalize to 0-1
    img = img / 255.0

    return img


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocess image for model input (matches training preprocessing)

    Args:
        image_path: Path to image file (DICOM, PNG, JPG, etc.)
        target_size: Target size tuple (height, width)

    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, H, W)
    """
    # Determine file type and load accordingly
    if image_path.lower().endswith('.dcm'):
        if not DICOM_AVAILABLE:
            raise ValueError("DICOM file provided but pydicom is not installed")
        img_uint8 = load_dicom_image(image_path)  # Already uint8 0-255
    else:
        img = load_image_file(image_path)  # Returns 0-1 float
        img_uint8 = (img * 255).astype(np.uint8)

    # Convert grayscale to RGB (3 channels) BEFORE resize
    if len(img_uint8.shape) == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)

    # Resize to target size using PIL (same as training)
    img_pil = Image.fromarray(img_uint8)
    img_pil = img_pil.resize(target_size)  # (width, height) for PIL
    img_array = np.array(img_pil)

    # Convert to torch tensor format (C, H, W) and normalize to 0-1
    # This matches training: img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def prepare_clinical_features(age, gender, height, weight, bmi):
    """
    Prepare clinical features tensor

    Args:
        age: Patient age (years)
        gender: Patient gender ("Male" or "Female")
        height: Patient height (cm)
        weight: Patient weight (kg)
        bmi: Body mass index

    Returns:
        torch.Tensor: Clinical features tensor (1, 1, 5)
    """
    # Convert gender to numeric (Male=0, Female=1)
    gender_numeric = 1.0 if gender == "Female" else 0.0

    # Create feature array: [AGE, Gender, Height, Weight, BMI]
    features = np.array([
        float(age),
        gender_numeric,
        float(height),
        float(weight),
        float(bmi)
    ], dtype=np.float32)

    # Convert to tensor with shape (1, 1, 5)
    features_tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)

    return features_tensor


def validate_inputs(age, gender, height, weight, bmi):
    """
    Validate clinical input parameters

    Args:
        age, gender, height, weight, bmi: Clinical parameters

    Returns:
        tuple: (is_valid, error_message)
    """
    from config import AGE_RANGE, HEIGHT_RANGE, WEIGHT_RANGE, BMI_RANGE, GENDER_OPTIONS

    # Age validation
    if not (AGE_RANGE[0] <= age <= AGE_RANGE[1]):
        return False, f"Age must be between {AGE_RANGE[0]} and {AGE_RANGE[1]} years"

    # Gender validation
    if gender not in GENDER_OPTIONS:
        return False, f"Gender must be one of {GENDER_OPTIONS}"

    # Height validation
    if not (HEIGHT_RANGE[0] <= height <= HEIGHT_RANGE[1]):
        return False, f"Height must be between {HEIGHT_RANGE[0]} and {HEIGHT_RANGE[1]} cm"

    # Weight validation
    if not (WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1]):
        return False, f"Weight must be between {WEIGHT_RANGE[0]} and {WEIGHT_RANGE[1]} kg"

    # BMI validation
    if not (BMI_RANGE[0] <= bmi <= BMI_RANGE[1]):
        return False, f"BMI must be between {BMI_RANGE[0]} and {BMI_RANGE[1]}"

    # Check if BMI is consistent with height and weight
    calculated_bmi = weight / ((height / 100) ** 2)
    if abs(calculated_bmi - bmi) > 0.5:  # Allow small tolerance
        return False, f"BMI ({bmi:.1f}) is inconsistent with height and weight (calculated BMI: {calculated_bmi:.1f})"

    return True, ""


def calculate_bmi(height_cm, weight_kg):
    """
    Calculate BMI from height and weight

    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms

    Returns:
        float: BMI value
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)
