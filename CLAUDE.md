# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About MM-CL

MM-CL (Multi-modality Contrastive Learning) is a research project for sarcopenia screening from hip X-rays and clinical information, published in MICCAI 2023. This repository has been **refactored from classification to regression** - specifically for ASMI (Appendicular Skeletal Muscle Index) regression prediction.

## Project Structure

The codebase follows a modular architecture optimized for regression tasks:

- `driver/` - Core training infrastructure
  - `reg_driver/` - **Regression-specific drivers and helpers** (renamed from cls_driver)
    - `RegHelper.py` - Main regression helper with CSV data loading and batch processing
    - `train.py` - Training script for ASMI regression
    - `config/reg_configuration.txt` - Regression configuration with MSE loss, Adam optimizer
  - `Config.py` - Configuration management system using ConfigParser
  - `base_train_helper.py` - Base training utilities with **MPS/CUDA/CPU device support**

- `models/` - Model definitions
  - `reg_models/resnet.py` - ResNet backbone and **ResNetFusionTextNetRegression** model
  - `BaseModel.py` - Base model class
  - `EMA.py` - Exponential Moving Average implementation

- `module/` - Neural network components
  - `backbone/` - Feature extraction backbones  
  - `head.py` - Regression heads (ResRegLessCNN)
  - `non_local.py` - Non-local attention mechanisms
  - `torchcam/` - GradCAM implementations

- `sarcopenia_data/` - **CSV-based data loading**
  - `SarcopeniaDataLoader.py` - CSV dataset classes for regression with DICOM image loading
  - `auto_augment.py` - Data augmentation strategies

- `commons/` - Shared utilities
  - `utils.py` - Common utility functions and regression metrics
  - `constant.py` - Constants including ASMI_COL, IMG_PATH_COL, TEXT_COLS

## Key Architecture Changes (Classification → Regression)

### Data Pipeline Transformation
- **Input**: CSV format (`data/data.csv`) instead of PTH files
- **Target**: ASMI continuous values instead of binary classification
- **Images**: DICOM files loaded via IMG_PATH column
- **Clinical features**: 5 features (AGE, Gender, Height, Weight, BMI)
- **Data split**: Last 100 samples for testing, remainder for K-fold training

### Model Architecture
- **ResNetFusionTextNetRegression**: Modified fusion model with single regression output
- **Multi-device support**: Automatic CUDA/MPS/CPU selection in base_train_helper
- **Loss function**: MSE loss instead of cross-entropy
- **Output**: Single continuous ASMI value (kg/m²)

### Training Framework
- **RegHelper class**: Handles CSV data loading, regression batch merging
- **Metrics**: MAE, MSE, R², Pearson correlation instead of classification metrics
- **Cross-validation**: 5-fold support for robust evaluation
- **Early stopping**: Based on validation loss with configurable patience

## Running the Code

### Training ASMI Regression Model
```bash
cd driver/reg_driver
python train.py --config-file ./config/reg_configuration.txt --gpu 0 --seed 666
```

### Testing Trained Models
```bash
cd driver/reg_driver
python test.py --config-file ./config/reg_configuration.txt --gpu 0 --load-best-epoch
```

### Generating Visualizations and Analysis
```bash
cd driver/reg_driver
# Generate publication-ready plots from training logs
python plot.py --log-dir ../../log/ASMI_Regression/ResNetFusionTextNetRegression/0_run_ASMI-Reg_*/

# For specific log directory with timestamp
python plot.py --log-dir ../../log/ASMI_Regression/ResNetFusionTextNetRegression/0_run_ASMI-Reg_2024-XX-XX_XX-XX-XX/
```

### Common Development Commands

#### Training with different configurations
```bash
# Train with specific GPU and random seed
python train.py --config-file ./config/reg_configuration.txt --gpu 0 --seed 42

# Train with CPU only (for testing)
python train.py --config-file ./config/reg_configuration.txt --gpu -1

# Train with different backbone
python train.py --config-file ./config/reg_configuration.txt --backbone resnet50

# Custom model override
python train.py --config-file ./config/reg_configuration.txt --model ResNetFusionTextNetRegression --backbone resnet18
```

#### Testing and evaluation
```bash
# Test trained model and generate predictions
python test.py --config-file ./config/reg_configuration.txt --gpu 0

# Test with specific model checkpoint
python test.py --config-file ./config/reg_configuration.txt --load-model-path /path/to/checkpoint
```

### Key Configuration Parameters
The regression configuration (`driver/reg_driver/config/reg_configuration.txt`) controls:
- **Model**: ResNetFusionTextNetRegression with ResNet34 backbone (default)
- **Data**: CSV-based loading with DICOM images (224x224 pixels)
- **Loss**: MSE loss function (alternatives: huber, mae, smooth_l1)
- **Optimizer**: Adam with 1e-4 learning rate, 1e-5 weight decay
- **Training**: 32 batch size, 5-fold cross-validation, early stopping (patience=7)
- **Learning Rate**: ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- **Advanced Features**: Contrastive learning, CAM enhancement, implant removal
- **Evaluation**: MAE, MSE, RMSE, R², Pearson correlation metrics

### Expected Data Format
- **CSV file**: `data/data.csv` with columns:
  - `ASMI(kg/m2)`: Target regression values
  - `IMG_PATH`: Paths to DICOM image files  
  - `編號`: Sample identifiers
  - Clinical features: `AGE`, `Gender(M:0,F:1)`, `身高cm`, `體重kg`, `BMI`
- **Image files**: DICOM format accessible via IMG_PATH column

## Development Notes

### Core Classes and Methods
- **RegHelper** (`driver/reg_driver/RegHelper.py`): `get_data_loader_csv()`, `merge_batch_regression()`, `get_test_data_loader_csv()`
- **SarcopeniaCSVDataSet** (`sarcopenia_data/SarcopeniaDataLoader.py`): CSV-based dataset with DICOM loading and clinical feature extraction
- **ResNetFusionTextNetRegression** (`models/reg_models/resnet.py`): Multi-modal fusion with regression head
- **Config** (`driver/Config.py`): ConfigParser-based configuration management with section interpolation

### Device Management
The system automatically detects and uses the best available device:
1. CUDA (if available and requested)
2. MPS (Apple Silicon)  
3. CPU (fallback)

### Key Dependencies
- PyTorch with torchvision for deep learning
- pandas for CSV data handling
- pydicom for DICOM image loading
- einops for tensor operations
- sklearn for regression metrics
- scipy for Pearson correlation

### Regression Metrics
The system evaluates using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)
- Pearson correlation coefficient

## Configuration Management

Uses ConfigParser with section-based organization and variable interpolation:
- `[Data]`: Dataset paths, image dimensions, CSV configuration (`data_path`, `csv_path`)
- `[Network]`: Model architecture, backbone selection (`model`, `backbone`)
- `[Optimizer]`: Learning rate, weight decay, algorithm parameters (`learning_rate`, `weight_decay`)
- `[Loss]`: Loss function selection and parameters (`loss_function`: mse, huber, mae, smooth_l1)
- `[Run]`: Training parameters, device selection, cross-validation (`n_epochs`, `train_batch_size`, `nfold`)
- `[Save]`: Output directories with variable interpolation (`save_dir`, `load_model_path`)
- `[Evaluation]`: Metrics selection and output formatting (`metrics`, `save_predictions`)

### Key Configuration Features
- **Variable interpolation**: `${Section:variable}` syntax for path construction
- **Multi-fold cross-validation**: Configurable via `nfold` parameter
- **Early stopping**: Controlled by `early_stopping` and `patience` parameters
- **Device selection**: Automatic fallback from CUDA → MPS → CPU

## Quick Development Setup

### Fast Validation Testing
For rapid development iteration, modify `driver/reg_driver/config/reg_configuration.txt`:
```ini
[Run]
n_epochs = 1          # Quick training validation
nfold = 2            # Reduced folds for speed
train_batch_size = 16 # Smaller batches if memory limited
```

### Development Workflow
1. **Quick Test**: Run with modified config (1 epoch, 2 folds)
2. **Debug Data**: Check `log/` directory for outputs and visualizations
3. **Full Training**: Restore `n_epochs = 50+` and `nfold = 5` for final runs
4. **Analysis**: Generated plots and CSV files appear in `log/${Data:data_name}/${Network:model}/${Run:gpu}_run_${Run:run_num}/`
5. **Visualization**: Use `python plot.py` to generate publication-ready plots
6. **Model Selection**: Compare different backbone options (resnet18/34/50/101)

### Advanced Configuration Options
For specialized training scenarios, modify key parameters:

```ini
# Implant removal (reduces bias towards metal implants)
[Data]
remove_implants = True
implant_threshold = 240
removal_strategy = gaussian_noise

# Contrastive learning enhancement
[Contrastive]
contrastive_learning = True
cam_enhancement = True
contrastive_beta = 0.01

# Backbone selection guide
[Network]
backbone = resnet18    # 512D features, fastest training
backbone = resnet34    # 512D features, best balance (default)
backbone = resnet50    # 2048D features, deeper architecture
backbone = resnet101   # 2048D features, best performance but slower
```

### Output Structure
Training generates comprehensive outputs:
- **Console logs**: `log/.../console_log.txt`
- **Model checkpoints**: `log/.../checkpoint/fold_X_best_model.pth`
- **Visualizations**: Training curves, scatter plots, residual analysis
- **CSV data**: Detailed predictions and metrics for further R analysis
- **AnalysisHelper**: Automated generation of publication-ready plots

### Multi-Modal Architecture Details
The core fusion model combines:
1. **ResNet18 backbone** → **Non-local blocks** → Visual features
2. **TextNet** (Conv1d + BatchNorm + SiLU) → Clinical features
3. **Self-attention fusion** → Combined representation → **ResRegLessCNN** → ASMI prediction

**Data Flow**: X-ray DICOM (224×224) + Clinical CSV (5 features) → Multi-modal fusion → ASMI regression (kg/m²)