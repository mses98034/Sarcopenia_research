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

## Git工作流程 (Git新手教學)

### 基本Git操作

#### 1. 檢查狀態和歷史
```bash
# 查看當前狀態
git status

# 查看提交歷史
git log --oneline

# 查看檔案變更
git diff
```

#### 2. 基本提交流程
```bash
# 加入檔案到暫存區
git add filename.py                # 加入單一檔案
git add commons/ models/           # 加入特定資料夾
git add .                          # 加入所有變更 (小心使用)

# 提交變更
git commit -m "描述你做了什麼變更"

# 推送到GitHub
git push origin main
```

#### 3. 撤銷操作
```bash
# 撤銷還未commit的變更
git checkout -- filename.py       # 撤銷單一檔案
git reset --hard                   # 撤銷所有未commit變更 (危險!)

# 撤銷已加入暫存區的檔案
git reset HEAD filename.py

# 修改上一次commit訊息
git commit --amend -m "新的commit訊息"
```

### GitHub Repository設置

#### 1. 創建GitHub Repository
1. 前往 https://github.com
2. 點擊右上角 "+" → "New repository"
3. Repository name: `MM-CL`
4. 設為Public (或Private，依需求)
5. **不要**勾選 "Add a README file" (因為已有檔案)
6. 點擊 "Create repository"

#### 2. 連接本地和遠端repository
```bash
# 加入GitHub remote (替換YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/MM-CL.git

# 推送初始commit
git branch -M main                 # 將master改名為main
git push -u origin main           # 推送並設置upstream

# 檢查remote設定
git remote -v
```

### Branch管理

#### 1. 創建和切換分支
```bash
# 創建新分支並切換
git checkout -b feature/new-model

# 或使用較新的語法
git switch -c feature/new-model

# 切換到現有分支
git checkout main
git switch main

# 查看所有分支
git branch -a
```

#### 2. 合併分支
```bash
# 切換到main分支
git checkout main

# 合併feature分支
git merge feature/new-model

# 刪除已合併的分支
git branch -d feature/new-model
```

### Issue管理

#### 1. 創建Issue
1. 前往GitHub repository頁面
2. 點擊 "Issues" tab
3. 點擊 "New issue"
4. 填寫標題和描述：
   ```
   標題: Fix ASMI regression model accuracy

   描述:
   ## 問題描述
   當前模型在驗證集上的R²分數過低 (<0.5)

   ## 預期結果
   提升R²分數至0.7以上

   ## 可能的解決方案
   - [ ] 調整learning rate
   - [ ] 嘗試不同backbone
   - [ ] 增加data augmentation
   ```
5. 指派Labels (bug, enhancement, documentation等)
6. 點擊 "Submit new issue"

#### 2. 關閉Issue
在commit訊息中使用關鍵字：
```bash
git commit -m "Fix model accuracy by adjusting hyperparameters

- Reduced learning rate to 1e-5
- Added dropout to prevent overfitting
- Improved R² score from 0.45 to 0.72

Fixes #1"
```

### Pull Request (PR) 工作流程

#### 1. 標準PR流程
```bash
# 1. 創建feature分支
git checkout -b feature/improve-accuracy

# 2. 進行程式碼修改
# ... 編輯檔案 ...

# 3. 提交變更
git add models/reg_models/resnet.py
git commit -m "Improve model accuracy with better hyperparameters"

# 4. 推送分支到GitHub
git push origin feature/improve-accuracy
```

#### 2. 在GitHub創建PR
1. 前往GitHub repository
2. 會看到 "Compare & pull request" 按鈕，點擊它
3. 填寫PR資訊：
   ```
   標題: Improve ASMI regression model accuracy

   描述:
   ## Changes
   - Adjusted learning rate from 1e-4 to 1e-5
   - Added dropout layer (p=0.3) to ResNet fusion
   - Implemented early stopping with patience=10

   ## Results
   - R² score improved from 0.45 to 0.72
   - MAE reduced from 1.2 to 0.8

   ## Testing
   - [x] Trained on full dataset (5-fold CV)
   - [x] Verified on test set
   - [x] All tests pass

   Closes #1
   ```
4. 選擇Reviewers (如有協作者)
5. 點擊 "Create pull request"

#### 3. 合併PR
1. 等待review (如果是個人專案可直接合併)
2. 在GitHub上點擊 "Merge pull request"
3. 選擇合併方式：
   - **Create a merge commit**: 保留分支歷史
   - **Squash and merge**: 將多個commit合併為一個
   - **Rebase and merge**: 重新排列commit歷史

### 日常開發工作流程

#### 1. 每日工作開始前
```bash
# 確保在main分支並更新到最新
git checkout main
git pull origin main

# 創建新的feature分支
git checkout -b feature/YYYY-MM-DD-task-description
```

#### 2. 開發過程中
```bash
# 經常提交小的變更
git add .
git commit -m "Add data preprocessing validation"

# 定期推送到遠端
git push origin feature/YYYY-MM-DD-task-description
```

#### 3. 完成功能後
```bash
# 確保code quality
python driver/reg_driver/train.py --config-file ./config/reg_configuration.txt --gpu 0  # 測試

# 最終提交
git add .
git commit -m "Complete feature implementation with tests"
git push origin feature/YYYY-MM-DD-task-description

# 在GitHub創建PR
```

### 常見問題解決

#### 1. 合併衝突
```bash
# 當git merge或git pull出現衝突時
git status                         # 查看衝突檔案

# 手動編輯衝突檔案，移除 <<<<<<< ======= >>>>>>> 標記
# 解決後：
git add conflicted_file.py
git commit -m "Resolve merge conflicts"
```

#### 2. 錯誤的commit
```bash
# 還沒push的情況下，撤銷上一個commit
git reset --soft HEAD~1            # 保留變更
git reset --hard HEAD~1            # 完全撤銷 (危險!)

# 已經push的情況，創建新的commit來修正
git revert HEAD
```

#### 3. 忘記創建分支就開始開發
```bash
# 創建分支並保留當前變更
git stash                          # 暫存變更
git checkout -b feature/fix-issue  # 創建分支
git stash pop                      # 恢復變更
```

### 實用Git別名設定
```bash
# 設置常用的git alias
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm commit
git config --global alias.lg "log --oneline --graph --all"

# 使用例子
git st                            # 等同於 git status
git lg                           # 美化的log顯示
```

### 專案特定工作流程

對於MM-CL專案，建議的分支命名規範：
- `feature/model-improvement` - 模型改進
- `feature/data-preprocessing` - 資料前處理
- `bugfix/training-crash` - 修復bug
- `experiment/new-backbone` - 實驗性功能

提交訊息格式建議：
```
類型: 簡短描述 (50字元內)

詳細說明 (如需要):
- 具體變更內容
- 實驗結果或性能改進
- 相關Issue編號

範例:
feat: Add ResNet50 backbone support

- Implemented ResNet50 in backbone module
- Updated configuration to support deeper architectures
- Improved R² score from 0.68 to 0.74 on validation set

Closes #15
```