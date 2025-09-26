#!/usr/bin/env python
"""
ASMI Regression Testing Script
Test trained models on the hold-out test set
"""

import sys
sys.path.extend(["../../", "../", "./"])
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from driver.Config import Configurable
from driver.cls_driver.RegHelper import RegHelper
from models import MODELS
from commons.device_utils import get_optimal_device, setup_device_config
from commons.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class ASMIRegressionTester:
    def __init__(self, args, extra_args):
        # Load configuration
        self.config = Configurable(args, extra_args)
        self.config.train = False
        
        # Override config with command line arguments
        if args.gpu is not None:
            self.config.gpu = args.gpu
        if args.batch_size is not None:
            self.config.test_batch_size = args.batch_size
        if args.model_path is not None:
            self.config.load_model_path = args.model_path
            
        # Setup device
        self.config = setup_device_config(self.config, gpu_id=args.gpu, verbose=True)
        self.device = self.config.device
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        criterions = {'mse': self.criterion}
        
        # Initialize helper
        self.reg_helper = RegHelper(criterions, self.config)
        
        print(f"Testing configuration loaded from: {args.config_file}")
        print(f"Model: {self.config.model}")
        print(f"Device: {self.config.device}")
        print(f"Model path: {self.config.load_model_path}")

    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = os.path.join(self.config.load_model_path, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Build model
        if self.config.model == 'ResNetFusionTextNetRegression':
            model = MODELS[self.config.model](
                backbone=self.config.backbone,
                n_channels=self.config.n_channels,
                use_pretrained=self.config.use_pretrained
            )
        else:
            raise ValueError(f"Unsupported model: {self.config.model}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            val_metrics = checkpoint['metrics'].get('val_metrics', {})
            if val_metrics:
                print(f"Validation metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        return model

    def test_model(self, model, test_loader, save_predictions=True):
        """Test model on test set"""
        model.eval()
        predictions = []
        targets = []
        sample_info = []
        total_loss = 0.0
        
        print("Testing model...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = self.reg_helper.move_batch_to_device(batch)
                
                # Forward pass
                text_included = self.config.use_text_features
                if hasattr(model, 'forward') and 'text_included' in model.forward.__code__.co_varnames:
                    outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=text_included)
                else:
                    outputs = model(batch['image_patch'])
                
                # Flatten outputs and targets
                outputs = outputs.squeeze()
                targets_batch = batch['image_asmi']
                
                # Calculate loss
                loss = self.criterion(outputs, targets_batch)
                total_loss += loss.item()
                
                # Store results
                batch_preds = outputs.detach().cpu().numpy()
                batch_targets = targets_batch.detach().cpu().numpy()
                batch_names = batch['image_name']
                batch_paths = batch['image_path']
                
                # Handle single sample batches
                if np.isscalar(batch_preds):
                    batch_preds = [batch_preds]
                    batch_targets = [batch_targets]
                
                predictions.extend(batch_preds)
                targets.extend(batch_targets)
                
                for i in range(len(batch_preds)):
                    sample_info.append({
                        'sample_id': batch_names[i] if i < len(batch_names) else f'sample_{len(sample_info)}',
                        'image_path': batch_paths[i] if i < len(batch_paths) else 'unknown',
                        'actual_asmi': batch_targets[i],
                        'predicted_asmi': batch_preds[i],
                        'absolute_error': abs(batch_targets[i] - batch_preds[i])
                    })
        
        avg_loss = total_loss / len(test_loader)
        
        # Calculate comprehensive metrics
        metrics = self.reg_helper.calculate_regression_metrics(targets, predictions)
        metrics['test_loss'] = avg_loss
        
        print(f"\nTest Results:")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"Pearson r: {metrics['pearson']:.4f} (p={metrics['pearson_p']:.4f})")
        
        # Save predictions if requested
        if save_predictions:
            self.save_test_results(sample_info, metrics, predictions, targets)
        
        return metrics, predictions, targets, sample_info

    def save_test_results(self, sample_info, metrics, predictions, targets):
        """Save test results to files"""
        # Create results directory
        results_dir = os.path.join(self.config.load_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame(sample_info)
        predictions_path = os.path.join(results_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        
        # Save metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(results_dir, 'test_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")
        
        # Generate plots
        self.generate_test_plots(predictions, targets, results_dir)
        
        return results_dir

    def generate_test_plots(self, predictions, targets, save_dir):
        """Generate comprehensive test result plots"""
        # Set seaborn style for beautiful plots
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams['figure.facecolor'] = 'white'

        # 1. Main regression plot with seaborn styling
        plt.figure(figsize=(18, 12))

        # Beautiful scatter plot with density and regression line
        plt.subplot(2, 3, 1)

        # Create scatter plot with seaborn
        sns.scatterplot(x=targets, y=predictions, alpha=0.7, s=60, color='steelblue', edgecolor='white', linewidth=0.5)

        # Perfect prediction line
        min_val, max_val = min(min(targets), min(predictions)), max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect prediction', alpha=0.8)

        # Regression line with seaborn
        sns.regplot(x=targets, y=predictions, scatter=False, color='darkblue', line_kws={'linewidth': 2})

        plt.xlabel('Actual ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.title('Predicted vs Actual ASMI Values', fontsize=14, fontweight='bold', pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=True)

        # Add correlation coefficient with better styling
        corr, p_val = pearsonr(targets, predictions)
        textstr = f'Pearson r = {corr:.3f}\np-value = {p_val:.3e}'
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                bbox=props, verticalalignment='top')
        
        # 2. Beautiful residual plot
        plt.subplot(2, 3, 2)
        residuals = np.array(predictions) - np.array(targets)
        sns.scatterplot(x=predictions, y=residuals, alpha=0.7, s=60, color='coral', edgecolor='white', linewidth=0.5)
        plt.axhline(y=0, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
        plt.xlabel('Predicted ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
        plt.title('Residual Analysis', fontsize=14, fontweight='bold', pad=15)
        
        # 3. Beautiful distribution comparison
        plt.subplot(2, 3, 3)
        sns.histplot(targets, alpha=0.7, bins=20, label='Actual', color='steelblue', kde=True)
        sns.histplot(predictions, alpha=0.7, bins=20, label='Predicted', color='darkorange', kde=True)
        plt.xlabel('ASMI Values (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=True)

        # 4. Beautiful error distribution
        plt.subplot(2, 3, 4)
        absolute_errors = np.abs(residuals)
        sns.histplot(absolute_errors, bins=20, alpha=0.8, color='crimson', kde=True)
        plt.axvline(np.mean(absolute_errors), color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean AE: {np.mean(absolute_errors):.3f}')
        plt.xlabel('Absolute Error (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Absolute Error Distribution', fontsize=14, fontweight='bold', pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        # 5. Beautiful Bland-Altman plot
        plt.subplot(2, 3, 5)
        mean_values = (np.array(targets) + np.array(predictions)) / 2
        differences = np.array(predictions) - np.array(targets)

        sns.scatterplot(x=mean_values, y=differences, alpha=0.7, s=60, color='mediumseagreen', edgecolor='white', linewidth=0.5)
        
        # Calculate limits of agreement
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff
        
        plt.axhline(mean_diff, color='darkred', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
        plt.axhline(upper_limit, color='darkred', linestyle='--', alpha=0.8, linewidth=1.5, label=f'+1.96 SD: {upper_limit:.3f}')
        plt.axhline(lower_limit, color='darkred', linestyle='--', alpha=0.8, linewidth=1.5, label=f'-1.96 SD: {lower_limit:.3f}')

        plt.xlabel('Mean of Actual and Predicted (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Difference (Predicted - Actual)', fontsize=12, fontweight='bold')
        plt.title('Bland-Altman Agreement Plot', fontsize=14, fontweight='bold', pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=True)

        # 6. Beautiful Q-Q plot for residuals
        plt.subplot(2, 3, 6)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
        plt.ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, 'test_results_comprehensive.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive plots saved to: {plot_path}")

        # 2. Create a special density scatter plot like the example
        self.plot_density_scatter(targets, predictions, save_dir)

        # 3. Individual metric plots
        self.plot_individual_metrics(targets, predictions, residuals, save_dir)

    def plot_density_scatter(self, targets, predictions, save_dir):
        """Create a beautiful density scatter plot like the provided example"""
        plt.figure(figsize=(10, 8))

        # Create joint plot with density
        g = sns.jointplot(x=targets, y=predictions, kind='scatter',
                         height=8, space=0, alpha=0.6, s=50,
                         marginal_kws=dict(bins=25, fill=True))

        # Add perfect prediction line
        min_val, max_val = min(min(targets), min(predictions)), max(max(targets), max(predictions))
        g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Perfect prediction')

        # Add regression line
        sns.regplot(x=targets, y=predictions, ax=g.ax_joint, scatter=False,
                   color='darkblue', line_kws={'linewidth': 2})

        # Styling
        g.ax_joint.set_xlabel('True ASMI (kg/m²)', fontsize=14, fontweight='bold')
        g.ax_joint.set_ylabel('Predicted ASMI (kg/m²)', fontsize=14, fontweight='bold')
        g.ax_joint.legend()

        # Add correlation and metrics
        corr, p_val = pearsonr(targets, predictions)
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))

        # Add text box with metrics
        textstr = f'Pearson r = {corr:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        g.ax_joint.text(0.05, 0.95, textstr, transform=g.ax_joint.transAxes, fontsize=12,
                       bbox=props, verticalalignment='top')

        # Save the plot
        density_path = os.path.join(save_dir, 'prediction_density_scatter.png')
        plt.savefig(density_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Density scatter plot saved to: {density_path}")

    def plot_individual_metrics(self, targets, predictions, residuals, save_dir):
        """Generate individual metric plots with seaborn styling"""
        # Beautiful error by prediction value plot
        plt.figure(figsize=(12, 6))
        absolute_errors = np.abs(residuals)
        sns.scatterplot(x=predictions, y=absolute_errors, alpha=0.7, s=60, color='orange', edgecolor='white', linewidth=0.5)
        plt.xlabel('Predicted ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Error (kg/m²)', fontsize=12, fontweight='bold')
        plt.title('Absolute Error vs Predicted Value', fontsize=14, fontweight='bold', pad=15)

        # Add trend line with seaborn
        sns.regplot(x=predictions, y=absolute_errors, scatter=False, color='darkred', line_kws={'linewidth': 2})

        plt.savefig(os.path.join(save_dir, 'error_vs_prediction.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Beautiful prediction confidence intervals
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(targets)
        sorted_targets = np.array(targets)[sorted_indices]
        sorted_predictions = np.array(predictions)[sorted_indices]

        plt.plot(sorted_targets, sorted_targets, 'r--', linewidth=2.5, label='Perfect prediction', alpha=0.8)
        sns.scatterplot(x=sorted_targets, y=sorted_predictions, alpha=0.7, s=50, color='steelblue', edgecolor='white', linewidth=0.5)

        # Calculate prediction intervals
        residuals_sorted = sorted_predictions - sorted_targets
        std_residual = np.std(residuals_sorted)

        plt.fill_between(sorted_targets,
                        sorted_targets - 1.96 * std_residual,
                        sorted_targets + 1.96 * std_residual,
                        alpha=0.3, color='lightblue', label='95% Prediction Interval')

        plt.xlabel('Actual ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted ASMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.title('Predictions with 95% Confidence Intervals', fontsize=14, fontweight='bold', pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(save_dir, 'prediction_intervals.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='ASMI Regression Testing')
    
    # Configuration
    parser.add_argument('--config-file', '-c', default='./config/reg_configuration.txt',
                        help='Path to configuration file')
    parser.add_argument('--model-path', '-m', default=None,
                        help='Path to trained model file')
    
    # Testing parameters
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Batch size for testing')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Output options
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save predictions to CSV')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')
    
    # Testing modes
    parser.add_argument('--text-only', action='store_true',
                        help='Use only text features')
    parser.add_argument('--image-only', action='store_true',
                        help='Use only image features')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args, extra_args = parser.parse_known_args()
    
    # Initialize tester
    tester = ASMIRegressionTester(args, extra_args)
    
    # Load model
    model = tester.load_model(args.model_path)
    
    # Get test data loader
    test_loader = tester.reg_helper.get_test_data_loader_csv(text_only=args.text_only)
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Test model
    metrics, predictions, targets, sample_info = tester.test_model(
        model, test_loader, save_predictions=args.save_predictions
    )
    
    print(f"\nTesting completed successfully!")
    
    if args.verbose:
        # Print detailed sample results
        print(f"\nSample predictions (first 10):")
        for i, info in enumerate(sample_info[:10]):
            print(f"  {info['sample_id']}: Actual={info['actual_asmi']:.3f}, "
                  f"Predicted={info['predicted_asmi']:.3f}, Error={info['absolute_error']:.3f}")

if __name__ == '__main__':
    main()