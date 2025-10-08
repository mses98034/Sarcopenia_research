"""
Configuration file for ASMI Prediction Web App
"""
import os
import torch

# ============================================
# Model Configuration
# ============================================

# Best model path (absolute path)
BEST_MODEL_PATH = "/Users/leo/Documents/MM-CL/log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/checkpoint/best_model.pth"

# Model architecture configuration
MODEL_NAME = "ResNetFusionAttentionNetRegression"
BACKBONE = "resnet34"
N_CHANNELS = 3
USE_PRETRAINED = False  # Not needed for inference
USE_TEXT_FEATURES = True

# ============================================
# Image Preprocessing Configuration
# ============================================

# Input image size for model
IMAGE_SIZE = (224, 224)

# Note: Training uses simple /255 normalization, NOT ImageNet mean/std
# These parameters are kept for reference but not used
MEAN = [0.485, 0.456, 0.406]  # Not used - training uses /255 only
STD = [0.229, 0.224, 0.225]    # Not used - training uses /255 only

# ============================================
# Clinical Parameters Configuration
# ============================================

# Valid ranges for input validation
AGE_RANGE = (18, 120)
HEIGHT_RANGE = (100, 250)  # cm
WEIGHT_RANGE = (30, 300)   # kg
BMI_RANGE = (10, 60)

# Gender options
GENDER_OPTIONS = ["Male", "Female"]

# ============================================
# Sarcopenia Classification Thresholds
# ============================================

# AWGS (Asian Working Group for Sarcopenia) criteria
SARCOPENIA_THRESHOLDS = {
    "Male": 7.0,    # kg/m²
    "Female": 5.4   # kg/m²
}

# ============================================
# Device Configuration
# ============================================

def get_device():
    """Automatically detect the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# ============================================
# Gradio UI Configuration
# ============================================

# App title and description
APP_TITLE = "ASMI Prediction from Hip X-ray"
APP_DESCRIPTION = """
## Appendicular Skeletal Muscle Index (ASMI) Prediction Tool

This application uses a deep learning model to predict ASMI from hip X-ray images and clinical parameters.

**How to use:**
1. Upload a hip X-ray image (DICOM, PNG, or JPG format)
2. Enter patient's clinical information
3. Click "Predict" to get ASMI estimation and sarcopenia risk assessment

**Note:** This tool is for research purposes only and should not be used for clinical decision-making without professional medical evaluation.
"""

# Example inputs for demonstration
EXAMPLE_INPUTS = [
    ["examples/example_xray.png", 65, "Male", 170, 65, 22.5],
    ["examples/example_xray.png", 72, "Female", 158, 52, 20.8],
]

# ============================================
# CAM Visualization Configuration
# ============================================

# Whether to generate CAM visualization
ENABLE_CAM = True

# CAM colormap
CAM_COLORMAP = "jet"  # Options: jet, viridis, plasma, inferno

# ============================================
# Project Paths
# ============================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to sys.path for importing modules
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
