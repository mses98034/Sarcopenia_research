#!/usr/bin/env python
"""
Simple Training Script for ASMI Regression
Based on original test.py structure with minimal modifications
"""

import sys
sys.path.extend(["../../", "../", "./"])
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import argparse
from driver.reg_driver.RegHelper import RegHelper
from driver.reg_driver.AnalysisHelper import AnalysisHelper
from driver.Config import Configurable
from tqdm import tqdm


class InfoNCELoss(nn.Module):
    """InfoNCE Loss for contrastive learning adapted for regression tasks"""

    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        """
        Args:
            features: [2N, D] tensor where N is batch size, 2N includes both augmented views
        """
        # features shape: [2N, D], 2N = 2 * batch_size
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels for positive pairs
        N = features.shape[0] // 2
        # For each sample i, its positive pair is at i+N (or i-N)
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(features.device)

        return self.criterion(similarity_matrix, labels)


def train_epoch_contrastive(model, train_loader, optimizer, criterion, device, config):
    """Train for one epoch with contrastive learning"""
    model.train()
    total_loss = 0.0
    total_regression_loss = 0.0
    total_contrastive_loss = 0.0
    predictions = []
    targets = []

    # Initialize contrastive loss function and weight factor
    contrastive_criterion = InfoNCELoss(temperature=config.contrastive_temperature).to(device)
    beta = config.contrastive_beta  # Weight factor for contrastive loss

    for batch in tqdm(train_loader, desc="Training with Contrastive Learning"):
        # Get paired data from contrastive mode dataset
        images1 = batch['image_patch_1'].to(device)
        images2 = batch['image_patch_2'].to(device)
        texts = batch['image_text'].to(device)
        targets_batch = batch['image_asmi'].to(device)

        # Combine two versions of images into 2N batch
        batch_images = torch.cat([images1, images2], dim=0)
        # Text data also needs to be duplicated to match images
        batch_texts = torch.cat([texts, texts], dim=0)

        optimizer.zero_grad()

        # Single forward pass, get both regression predictions and contrastive features
        outputs, z_i = model(batch_images, text=batch_texts, text_included=True)

        # --- Calculate Combined Loss ---
        # 1. Regression loss (MSE): only for first version of images
        regression_outputs = outputs[:images1.size(0)].squeeze()
        if regression_outputs.dim() == 0:
            regression_outputs = regression_outputs.unsqueeze(0)
        if targets_batch.dim() == 0:
            targets_batch = targets_batch.unsqueeze(0)
        loss_regression = criterion(regression_outputs, targets_batch)

        # 2. Contrastive loss: for all projection features z_i
        loss_contrastive = contrastive_criterion(z_i)

        # 3. Combined loss: weighted sum
        combined_loss = loss_regression + beta * loss_contrastive

        # Backward pass based on combined loss
        combined_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)

        optimizer.step()

        total_loss += combined_loss.item()
        total_regression_loss += loss_regression.item()
        total_contrastive_loss += loss_contrastive.item()

        # Convert to numpy for metrics calculation
        pred_np = regression_outputs.detach().cpu().numpy()
        target_np = targets_batch.detach().cpu().numpy()
        if pred_np.ndim == 0:
            pred_np = np.array([pred_np])
        if target_np.ndim == 0:
            target_np = np.array([target_np])
        predictions.extend(pred_np)
        targets.extend(target_np)

    avg_loss = total_loss / len(train_loader)
    avg_reg_loss = total_regression_loss / len(train_loader)
    avg_cont_loss = total_contrastive_loss / len(train_loader)

    # Calculate regression metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if len(set(targets)) > 1 else 0.0

    metrics = {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'regression_loss': avg_reg_loss,
        'contrastive_loss': avg_cont_loss
    }

    return avg_loss, metrics


def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        for key in ['image_patch', 'image_asmi', 'image_text']:
            if key in batch:
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'forward') and 'text_included' in model.forward.__code__.co_varnames:
            outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=True)
        else:
            outputs = model(batch['image_patch'])
        
        # Calculate loss
        outputs = outputs.squeeze()
        targets_batch = batch['image_asmi']
        # Ensure outputs and targets have compatible shapes
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        if targets_batch.dim() == 0:
            targets_batch = targets_batch.unsqueeze(0)
        loss = criterion(outputs, targets_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping using configuration
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        # Convert to numpy and ensure 1D for extend operation
        pred_np = outputs.detach().cpu().numpy()
        target_np = targets_batch.detach().cpu().numpy()
        if pred_np.ndim == 0:
            pred_np = np.array([pred_np])
        if target_np.ndim == 0:
            target_np = np.array([target_np])
        predictions.extend(pred_np)
        targets.extend(target_np)
    
    avg_loss = total_loss / len(train_loader)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if len(set(targets)) > 1 else 0.0

    return avg_loss, {'mae': mae, 'mse': mse, 'r2': r2}

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    uids = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for key in ['image_patch', 'image_asmi', 'image_text']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'text_included' in model.forward.__code__.co_varnames:
                outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=True)
            else:
                outputs = model(batch['image_patch'])
            
            outputs = outputs.squeeze()
            targets_batch = batch['image_asmi']
            # Ensure outputs and targets have compatible shapes
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if targets_batch.dim() == 0:
                targets_batch = targets_batch.unsqueeze(0)
            loss = criterion(outputs, targets_batch)
            
            total_loss += loss.item()
            # Convert to numpy and ensure 1D for extend operation
            pred_np = outputs.detach().cpu().numpy()
            target_np = targets_batch.detach().cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            if target_np.ndim == 0:
                target_np = np.array([target_np])
            predictions.extend(pred_np)
            targets.extend(target_np)
            uids.extend(batch['image_uid'])
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if len(set(targets)) > 1 else 0.0
    try:
        pearson_r, _ = pearsonr(targets, predictions)
    except:
        pearson_r = 0.0
    
    return avg_loss, {'mae': mae, 'mse': mse, 'r2': r2, 'pearson': pearson_r}, predictions, targets, uids

def print_cv_summary(all_fold_results):
    """Print cross-validation summary statistics"""
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # Extract metrics from all folds
    val_losses = [result['best_val_loss'] for result in all_fold_results]
    mae_scores = [result['final_val_metrics']['mae'] for result in all_fold_results]
    r2_scores = [result['final_val_metrics']['r2'] for result in all_fold_results]
    pearson_scores = [result['final_val_metrics']['pearson'] for result in all_fold_results]
    
    # Calculate statistics
    print(f"Validation Loss - Mean: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"MAE            - Mean: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"R²             - Mean: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Pearson r      - Mean: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    
    print("\\nPer-fold Results:")
    print("Fold | Val Loss | MAE     | R²      | Pearson")
    print("-" * 45)
    for i, result in enumerate(all_fold_results):
        print(f"{i+1:4d} | {result['best_val_loss']:8.4f} | {result['final_val_metrics']['mae']:7.4f} | {result['final_val_metrics']['r2']:7.4f} | {result['final_val_metrics']['pearson']:7.4f}")




def main_train(config, seed=111):
    """Main training function - similar to test.py main function"""
    
    # Setup loss function using configuration
    if config.loss_function == 'mse':
        criterion = nn.MSELoss()
    elif config.loss_function == 'huber':
        criterion = nn.HuberLoss(delta=config.huber_delta)
    elif config.loss_function == 'mae':
        criterion = nn.L1Loss()
    elif config.loss_function == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")

    criterions = {config.loss_function: criterion}
    
    # Initialize helper
    reg_help = RegHelper(criterions, config)
    reg_help.move_to_cuda()  # This now supports MPS too
    device = reg_help.equipment
    
    # Log training configuration
    config_msg = f"=== TRAINING CONFIGURATION ===\nDevice: {device}\nData name: {reg_help.config.data_name}\nImage size: {reg_help.config.patch_x}\nRandom seed: {seed}"
    reg_help.log.write(config_msg + "\n")
    reg_help.log.write(f"Model: {config.model}\n")
    reg_help.log.write(f"Backbone: {config.backbone}\n")
    reg_help.log.write(f"Use pretrained: {config.use_pretrained}\n")
    reg_help.log.write(f"Loss function: {config.loss_function}\n")
    reg_help.log.write(f"Optimizer: {config.learning_algorithm}\n")
    reg_help.log.write(f"Learning rate: {config.learning_rate}\n")
    reg_help.log.write(f"Scheduler patience: {config.scheduler_patience}\n")
    reg_help.log.write(f"Early stopping patience: {config.patience}\n")
    reg_help.log.write(f"Number of folds: {config.nfold}\n")
    reg_help.log.write(f"Epochs: {config.epochs}\n")
    reg_help.log.write(f"Batch size: {config.train_batch_size}\n")
    reg_help.log.write(f"Data augmentation: AutoAugment enabled for training\n")
    reg_help.log.flush()
    
    # Build model
    model = reg_help.create_model().to(device)
    
    # Setup optimizer using configuration
    from driver import OPTIM
    optimizer_class = OPTIM[config.learning_algorithm]
    optimizer_kwargs = {
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay
    }

    # Add algorithm-specific parameters
    if config.learning_algorithm == 'adam':
        optimizer_kwargs.update({
            'eps': config.epsilon,
            'betas': (config.beta1, config.beta2)
        })
    elif config.learning_algorithm == 'sgd':
        # Add momentum for SGD (default 0.9 if not in config)
        optimizer_kwargs['momentum'] = getattr(config, 'momentum', 0.9)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    print(f"Model: {config.model} | Backbone: {config.backbone} | Pretrained: {config.use_pretrained}")
    print(f"Using optimizer: {config.learning_algorithm.upper()} with params: {optimizer_kwargs}")
    print(f"Data augmentation: AutoAugment enabled for training")
    
    # Setup scheduler using configuration
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lrate
    )
    print(f"Scheduler: ReduceLROnPlateau (factor={config.scheduler_factor}, patience={config.scheduler_patience}, min_lr={config.min_lrate})")
    
    # 5-fold cross-validation training
    all_fold_results = []
    max_patience = config.patience

    # Track overall best model across all folds
    overall_best_loss = float('inf')
    overall_best_model_state = None
    overall_best_fold = -1
    
    cv_start_msg = f"=== STARTING {config.nfold}-FOLD CROSS-VALIDATION ==="
    reg_help.log.write(cv_start_msg + "\n")
    reg_help.log.flush()
    
    for fold in range(config.nfold):
        fold_header = f"{'='*60}FOLD {fold + 1}/{config.nfold} - {config.backbone.upper()}{'='*60}"
        print(fold_header)
        reg_help.log.write(fold_header + "\\n")
        reg_help.log.flush()
        
        # Get data loaders for current fold
        train_loader, val_loader = reg_help.get_data_loader_csv(fold=fold, seed=seed)
        
        # Reset model for each fold
        if fold > 0:
            # Clean up previous model memory
            del model
            torch.cuda.empty_cache()  # Free GPU memory

            model = reg_help.create_model().to(device)
            # Setup optimizer using configuration for new fold
            optimizer_class = OPTIM[config.learning_algorithm]
            optimizer_kwargs = {
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay
            }

            # Add algorithm-specific parameters
            if config.learning_algorithm == 'adam':
                optimizer_kwargs.update({
                    'eps': config.epsilon,
                    'betas': (config.beta1, config.beta2)
                })
            elif config.learning_algorithm == 'sgd':
                optimizer_kwargs['momentum'] = getattr(config, 'momentum', 0.9)

            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.min_lrate
            )
        
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_epoch_metrics = None
        patience_counter = 0
        fold_history = []
        
        for epoch in range(1, config.epochs + 1):
            # Train - check if contrastive learning is enabled
            contrastive_learning = getattr(config, 'contrastive_learning', False)
            if contrastive_learning:
                train_loss, train_metrics = train_epoch_contrastive(model, train_loader, optimizer, criterion, device, config)
            else:
                train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
            
            # Validate
            val_loss, val_metrics, _, _, _ = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # Track scheduler state
            if hasattr(scheduler, 'num_bad_epochs'):
                bad_epochs = scheduler.num_bad_epochs
            else:
                bad_epochs = 'N/A'

            # Log LR changes
            if old_lr != new_lr:
                reg_help.log.write(f"  📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e} (bad epochs: {bad_epochs}/{config.scheduler_patience})\n")
            elif epoch % config.printfreq == 0:
                reg_help.log.write(f"  📊 Scheduler state: bad epochs {bad_epochs}/{config.scheduler_patience}, current LR: {new_lr:.2e}\n")
            
            fold_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            
            # Print progress
            if epoch % config.printfreq == 0 or epoch == config.epochs:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_msg = f"Epoch {epoch}/{config.epochs} (LR: {current_lr:.2e})"
                train_msg = f"  Train - Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}"
                val_msg = f"  Val   - Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}, Pearson: {val_metrics['pearson']:.4f}"

                combined_msg = f"{epoch_msg}\n{train_msg}\n{val_msg}"
                reg_help.log.write(combined_msg + "\n")
                # Check validation loss improvement
                if epoch > 1:
                    prev_val_loss = fold_history[-2]['val_loss'] if len(fold_history) >= 2 else float('inf')
                    if val_loss < prev_val_loss:
                        reg_help.log.write(f"  ✅ Val loss improved: {prev_val_loss:.4f} → {val_loss:.4f}\n")
                    else:
                        reg_help.log.write(f"  ⚠️  Val loss increased: {prev_val_loss:.4f} → {val_loss:.4f}\n")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_epoch_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics.copy(),
                    'val_metrics': val_metrics.copy()
                }
                patience_counter = 0
                # Save best model state for this fold (in memory only)
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                early_stop_msg = f"Early stopping at epoch {epoch}"
                reg_help.log.write(f"{early_stop_msg}\n")
                reg_help.log.flush()
                break
        
        # Get final predictions for scatter plot
        model.load_state_dict(best_model_state)
        _, final_val_metrics, final_predictions, final_targets, final_uids = validate_epoch(model, val_loader, criterion, device)

        # Store fold results
        fold_result = {
            'fold': fold,
            'best_val_loss': best_val_loss,
            'final_val_metrics': final_val_metrics,
            'final_predictions': final_predictions,
            'final_targets': final_targets,
            'final_uids': final_uids, 
            'history': fold_history
        }
        all_fold_results.append(fold_result)
        
        fold_complete_msg = f"Fold {fold + 1} completed - Best Val Loss: {best_val_loss:.4f}"
        # Display metrics from the actual best epoch, not recomputed metrics
        if best_epoch_metrics:
            best_val_metrics = best_epoch_metrics['val_metrics']
            fold_metrics_msg = f"Best metrics - Epoch {best_epoch} - Loss: {best_val_loss:.4f}, MAE: {best_val_metrics['mae']:.4f}, R²: {best_val_metrics['r2']:.4f}, Pearson: {best_val_metrics['pearson']:.4f}"
        else:
            fold_metrics_msg = f"Best metrics - Epoch {best_epoch} - Loss: {best_val_loss:.4f}"

        fold_summary = f"{fold_complete_msg}\n{fold_metrics_msg}"
        reg_help.log.write(fold_summary + "\n")
        reg_help.log.flush()

        # Update overall best model if this fold is better
        if best_val_loss < overall_best_loss:
            overall_best_loss = best_val_loss
            overall_best_model_state = best_model_state.copy()
            overall_best_fold = fold
            print(f"🏆 New overall best model found in fold {fold + 1} with val_loss: {best_val_loss:.4f}")
    
    # Print cross-validation summary
    print_cv_summary(all_fold_results)

    # Save overall best model
    if overall_best_model_state is not None:
        best_model_path = os.path.join(config.save_model_path, 'best_model.pth')
        torch.save(overall_best_model_state, best_model_path)
        print(f"🏆 Overall best model saved to: {best_model_path}")
        print(f"   Best fold: {overall_best_fold + 1}, Val Loss: {overall_best_loss:.4f}")
    else:
        print("⚠️ Warning: No best model found to save")

    # Initialize analysis helper and save CSV data
    print("\n📊 Saving training data to CSV files...")
    analysis_helper = AnalysisHelper(config)
    analysis_helper.save_training_csv_data(all_fold_results, config.save_dir, reg_help, best_fold_idx=overall_best_fold)

    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved to: {config.save_dir}")
    print(f"📊 CSV data saved to: {config.save_dir}/csv_data/")
    print(f"🏆 Best model saved to: {config.save_dir}/checkpoint/best_model.pth")
    print(f"\n💡 To generate plots and visualizations, run:")
    print(f"   python plot.py --log-dir {config.save_dir}")

    return all_fold_results

if __name__ == '__main__':
    # Setup
    gpu = torch.cuda.is_available()
    print("GPU available:", gpu)
    torch.backends.cudnn.benchmark = True
    
    # Argument parser - based on test.py structure
    argparser = argparse.ArgumentParser(description='ASMI Regression Training')
    argparser.add_argument('--config-file', default='./config/reg_configuration.txt',
                          help='Path to configuration file')
    argparser.add_argument('--use-cuda', action='store_true', default=True,
                          help='Use CUDA if available')
    argparser.add_argument('--train', action='store_true', default=True,
                          help='Training mode')
    argparser.add_argument('--gpu', help='GPU device ID', default='0')
    argparser.add_argument('--gpu-count', help='Number of GPUs', default='1')
    argparser.add_argument('--run-num', help='Run identifier', default="ASMI-Reg")
    argparser.add_argument('--ema-decay', help='EMA decay rate', default="0.99")
    argparser.add_argument('--seed', help='Random seed', default=666, type=int)
    argparser.add_argument('--model', help='Model name', default="ResNetFusionTextNetRegression")
    argparser.add_argument('--backbone', help='Backbone name', default="resnet18")
    
    args, extra_args = argparser.parse_known_args()
    
    # Load configuration
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)
    
    # Set configuration flags
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    
    print("GPU using status:", config.use_cuda)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Start training
    main_train(config, seed=args.seed)