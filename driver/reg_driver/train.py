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
from driver.reg_driver.FoldResultManager import FoldResultManager
from sarcopenia_data.GlobalImageCache import GlobalImageCache
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
        outputs, z_i = model(batch_images, text=batch_texts, text_included=config.use_text_features)

        # --- Calculate Combined Loss ---
        # 1. Regression loss : only for first version of images
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
            outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=config.use_text_features)
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

def validate_epoch(model, val_loader, criterion, device, config):
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
                outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=config.use_text_features)
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

def print_cv_summary(all_fold_results, reg_help):
    """Print cross-validation summary statistics and write to log"""
    summary_lines = []
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("CROSS-VALIDATION SUMMARY")
    summary_lines.append(f"{'='*80}")

    # Extract metrics from all folds
    val_losses = [result['best_val_loss'] for result in all_fold_results]
    mae_scores = [result['final_val_metrics']['mae'] for result in all_fold_results]
    r2_scores = [result['final_val_metrics']['r2'] for result in all_fold_results]
    pearson_scores = [result['final_val_metrics']['pearson'] for result in all_fold_results]

    # Calculate statistics
    summary_lines.append(f"Validation Loss - Mean: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    summary_lines.append(f"MAE            - Mean: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    summary_lines.append(f"R2             - Mean: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    summary_lines.append(f"Pearson r      - Mean: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")

    summary_lines.append("\\nPer-fold Results:")
    summary_lines.append("Fold | Val Loss | MAE     | R2      | Pearson")
    summary_lines.append("-" * 45)
    for i, result in enumerate(all_fold_results):
        summary_lines.append(f"{i+1:4d} | {result['best_val_loss']:8.4f} | {result['final_val_metrics']['mae']:7.4f} | {result['final_val_metrics']['r2']:7.4f} | {result['final_val_metrics']['pearson']:7.4f}")

    # Print to console and write to log
    for line in summary_lines:
        print(line)
        reg_help.log.write(line + "\n")

    reg_help.log.flush()




def main_train(config, seed=111):
    """Main training function - similar to test.py main function"""
    
    from driver.reg_driver.losses import PearsonLoss, CCCLoss

    # Setup loss function using configuration
    if config.loss_function == 'mse':
        criterion = nn.MSELoss()
    elif config.loss_function == 'huber':
        criterion = nn.HuberLoss(delta=config.huber_delta)
    elif config.loss_function == 'mae':
        criterion = nn.L1Loss()
    elif config.loss_function == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif config.loss_function == 'pearson':
        criterion = PearsonLoss()
    elif config.loss_function == 'ccc':
        criterion = CCCLoss()
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

    if config.learning_algorithm in ['adam', 'adamw']:
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
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

    # Setup scheduler using configuration
    if config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,  # T_max is often the total number of epochs
            eta_min=config.min_lrate
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={config.epochs}, min_lr={config.min_lrate})")
    else:  # Default to plateau
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

    # === Initialize fold result manager for memory optimization ===
    fold_manager = FoldResultManager(config.save_dir)

    # === Initialize global image cache for K-fold optimization ===
    global_cache = None
    if config.use_global_image_cache:
        print("\n" + "="*60)
        print("Initializing Global Image Cache")
        print("="*60)
        reg_help.log.write("\n=== GLOBAL IMAGE CACHE INITIALIZATION ===\n")

        # Load complete training data (before K-fold split)
        from sarcopenia_data.SarcopeniaDataLoader import load_csv_data
        train_csv_path = os.path.join(config.data_path, config.train_filename)
        import pandas as pd
        all_train_data = pd.read_csv(train_csv_path)
        all_train_data.columns = [col.strip() for col in all_train_data.columns]

        # Create global cache
        global_cache = GlobalImageCache(
            csv_data=all_train_data,
            input_size=(config.patch_x, config.patch_y),
            load_images=True
        )

        cache_summary = global_cache.get_summary()
        cache_msg = f"Global cache ready: {cache_summary['cached_images']} images ({cache_summary['memory_mb']:.1f} MB)"
        print(cache_msg)
        print("="*60 + "\n")
        reg_help.log.write(f"Global cache: {cache_summary['cached_images']} images, {cache_summary['memory_mb']:.1f} MB\n")
        reg_help.log.flush()
    else:
        print("\nGlobal image cache is disabled")
        print("   Images will be loaded from disk for each batch (slower)")
        print("   To enable: set 'use_global_image_cache = True' in config\n")
        reg_help.log.write("Global image cache: Disabled\n")
        reg_help.log.flush()

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

        # Get data loaders for current fold (pass global cache)
        train_loader, val_loader = reg_help.get_data_loader_csv(fold=fold, seed=seed, shared_cache=global_cache)
        
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
            if config.learning_algorithm in ['adam', 'adamw']:
                optimizer_kwargs.update({
                    'eps': config.epsilon,
                    'betas': (config.beta1, config.beta2)
                })
            elif config.learning_algorithm == 'sgd':
                optimizer_kwargs['momentum'] = getattr(config, 'momentum', 0.9)

            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            if config.scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lrate)
            else: # Default to plateau
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
            val_loss, val_metrics, _, _, _ = validate_epoch(model, val_loader, criterion, device, config)
            
            # Update scheduler based on its type
            old_lr = optimizer.param_groups[0]['lr']
            if config.scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:  # For cosine or other epoch-based schedulers
                scheduler.step()
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
                # Check validation loss improvement (compare with best, not previous)
                if epoch > 1:
                    if val_loss < best_val_loss:
                        reg_help.log.write(f"  ✅ Val loss improved: {best_val_loss:.4f} → {val_loss:.4f} (new best!)\n")
                    else:
                        reg_help.log.write(f"  ⚠️  Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})\n")
            
            # Early stopping, only if enabled in config
            if config.early_stopping:
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
            # If early stopping is disabled, still track the best model from the entire run
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_epoch_metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_metrics': train_metrics.copy(),
                        'val_metrics': val_metrics.copy()
                    }
                    # Save best model state for this fold (in memory only)
                    best_model_state = model.state_dict().copy()
        
        # Get final predictions for scatter plot
        model.load_state_dict(best_model_state)
        _, final_val_metrics, final_predictions, final_targets, final_uids = validate_epoch(model, val_loader, criterion, device, config)

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

        # === Use FoldResultManager to offload large data to disk ===
        fold_metadata = fold_manager.save_fold_result(fold, fold_result)
        all_fold_results.append(fold_metadata)  # Only store lightweight metadata

        # --- TUNING REPORT ---
        # Print a machine-readable report for the tuning script to capture
        try:
            # Use the metrics from the best epoch for consistency
            report_data = {
                "fold": fold + 1,
                "pearson": best_epoch_metrics['val_metrics']['pearson'],
                "val_loss": best_epoch_metrics['val_loss']
            }
            import json
            print(f"[TUNING_REPORT] {json.dumps(report_data)}")
        except (KeyError, ImportError, TypeError):
            # Fail silently if keys are not found or json is not available
            pass


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

        # Update overall best model if this fold is better (BEFORE deleting best_model_state!)
        if best_val_loss < overall_best_loss:
            overall_best_loss = best_val_loss
            overall_best_model_state = best_model_state.copy()
            overall_best_fold = fold
            reg_help.log.write(
                f"🏆 New overall best model found in fold {fold + 1}\n"
                f"   Val Loss: {best_val_loss:.4f} | MAE: {best_epoch_metrics['val_metrics']['mae']:.4f} | "
                f"R²: {best_epoch_metrics['val_metrics']['r2']:.4f} | Pearson: {best_epoch_metrics['val_metrics']['pearson']:.4f}\n"
            )

        # Explicitly free memory (AFTER saving best model)
        del final_predictions, final_targets, final_uids, best_model_state
        torch.cuda.empty_cache()
    
    # Print cross-validation summary
    print_cv_summary(all_fold_results, reg_help)

    # Save overall best model
    if overall_best_model_state is not None:
        best_model_path = os.path.join(config.save_model_path, 'best_model.pth')
        torch.save(overall_best_model_state, best_model_path)
        reg_help.log.write(f"🏆 Overall best model saved to: {best_model_path}\n")
        reg_help.log.write(f"   Best fold: {overall_best_fold + 1}, Val Loss: {overall_best_loss:.4f}\n")
    else:
        reg_help.log.write("⚠️ Warning: No best model found to save\n")

    # === Load complete fold results from disk for analysis ===
    print("\n Preparing data for CSV generation...")
    reg_help.log.write("\n📊 Loading fold results from disk for analysis...\n")
    complete_fold_results = fold_manager.load_all_results()

    # Initialize analysis helper and save CSV data
    print(" Saving training data to CSV files...")
    reg_help.log.write("\n📊 Saving training data to CSV files...\n")
    analysis_helper = AnalysisHelper(config)
    analysis_helper.save_training_csv_data(complete_fold_results, config.save_dir, reg_help, best_fold_idx=overall_best_fold)

    # Clean up temporary fold data files
    fold_manager.cleanup()

    reg_help.log.write(f"\n🎉 Training completed!\n")
    reg_help.log.write(f"📁 Results saved to: {config.save_dir}\n")
    reg_help.log.write(f"📊 CSV data saved to: {config.save_dir}/csv_data/\n")
    reg_help.log.write(f"🏆 Best model saved to: {config.save_dir}/checkpoint/best_model.pth\n")
    reg_help.log.write(f"\n💡 To generate plots and visualizations, run:\n")
    reg_help.log.write(f"   python plot.py --log-dir {config.save_dir}\n")

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
    argparser.add_argument('--model', help='Model name', default=None)
    argparser.add_argument('--backbone', help='Backbone name', default=None)
    
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