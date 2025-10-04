#!/usr/bin/env python
"""
Independent plotting script for ASMI regression analysis
Reads CSV files from training logs and generates visualizations
"""

import sys
sys.path.extend(["../", "./"])
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Configure safe fonts (English only)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
plt.rcParams['font.size'] = 10  # Set default font size
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score

# Configure modern, readable typography for all plots
plt.rcParams.update({
    'font.family': ['sans-serif'],  # Use system default sans-serif font
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 22,
    'axes.titleweight': 'semibold',
    'axes.labelweight': 'semibold',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
})
import torch
import torch.nn.functional as F
import random

# --- Dynamic Project Module Imports ---
from driver.Config import Configurable
from models import MODELS
from module.torchcam.methods import SmoothGradCAMpp
from driver import transform_test
from commons.constant import TEXT_COLS
import cv2
import pydicom

class TrainingAnalyzer:
    def __init__(self, log_dir):
        """
        Initialize analyzer with log directory

        Args:
            log_dir: Path to training log directory containing csv_data/
        """
        self.log_dir = log_dir
        self.csv_dir = os.path.join(log_dir, 'csv_data')
        self.plots_dir = os.path.join(log_dir, 'plots')
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoint')

        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)

        # Load configuration file
        config_path = os.path.join(self.log_dir, 'configuration.txt')
        if os.path.exists(config_path):
            config_args = argparse.Namespace(config_file=config_path, train=False)
            self.config = Configurable(config_args, [])
            print(f"✅ Loaded configuration from: {config_path}")
        else:
            print(f"⚠️ Warning: configuration.txt not found. Some plot features may be limited.")
            self.config = None

        # Load all CSV data
        self.data = self.load_csv_data()

    def load_csv_data(self):
        """Load the 3 core CSV files into a dictionary"""
        data = {}

        csv_files = {
            'training_metrics': 'training_metrics.csv',
            'validation_predictions': 'validation_predictions.csv',
            'patient_data': 'patient_data.csv'
        }

        for key, filename in csv_files.items():
            file_path = os.path.join(self.csv_dir, filename)
            if os.path.exists(file_path):
                data[key] = pd.read_csv(file_path)
                print(f"✅ Loaded {filename}: {len(data[key])} rows")
            else:
                print(f"⚠️ Warning: {filename} not found")
                data[key] = None

        return data

    def plot_training_curves(self):
        """Generate aggregated learning curves with annotations for best performance."""
        if self.data['training_metrics'] is None:
            print("❌ Cannot plot training curves: training_metrics.csv not found")
            return

        print("📊 Generating enhanced aggregated learning curves with annotations...")

        df = self.data['training_metrics']

        # --- Dynamic Labels from Config ---
        if self.config and hasattr(self.config, 'loss_function'):
            loss_name = self.config.loss_function.upper()
        else:
            loss_name = 'Loss'  # Fallback if config is not available

        # --- Data Aggregation ---
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.6)
        plt.rcParams['figure.facecolor'] = 'white'

        agg_data = df.groupby('epoch').agg({
            'train_loss': ['mean', 'std'], 'val_loss': ['mean', 'std'],
            'train_mae': ['mean', 'std'], 'val_mae': ['mean', 'std'],
            'train_r2': ['mean', 'std'], 'val_r2': ['mean', 'std'],
            'val_pearson': ['mean', 'std']
        }).reset_index()
        agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_data.columns.values]

        # --- Find Best Performing Points for Annotation ---
        # Best validation loss (minimum)
        best_loss_idx = agg_data['val_loss_mean'].idxmin()
        best_loss_row = agg_data.loc[best_loss_idx]
        best_loss_epoch = int(best_loss_row['epoch'])
        best_loss_val = best_loss_row['val_loss_mean']

        # Best validation Pearson (maximum)
        best_pearson_idx = agg_data['val_pearson_mean'].idxmax()
        best_pearson_row = agg_data.loc[best_pearson_idx]
        best_pearson_epoch = int(best_pearson_row['epoch'])
        best_pearson_val = best_pearson_row['val_pearson_mean']

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Aggregated Learning Curves (5-Fold Cross-Validation)', fontsize=24, fontweight='semibold')

        # 1. Training/Validation Loss
        ax = axes[0, 0]
        ax.plot(agg_data['epoch'], agg_data['train_loss_mean'], color='steelblue', label='Training Loss', linewidth=2.5)
        ax.fill_between(agg_data['epoch'], agg_data['train_loss_mean'] - agg_data['train_loss_std'], agg_data['train_loss_mean'] + agg_data['train_loss_std'], color='steelblue', alpha=0.2)
        ax.plot(agg_data['epoch'], agg_data['val_loss_mean'], color='darkorange', label='Validation Loss', linewidth=2.5, linestyle='--')
        ax.fill_between(agg_data['epoch'], agg_data['val_loss_mean'] - agg_data['val_loss_std'], agg_data['val_loss_mean'] + agg_data['val_loss_std'], color='darkorange', alpha=0.2)
        
        # Annotation for best validation loss
        ax.plot(best_loss_epoch, best_loss_val, 'o', color='red', markersize=10, label=f'Best Val Loss: {best_loss_val:.4f}')
        ax.annotate(f'Best: {best_loss_val:.4f}\n@ Epoch {best_loss_epoch}',
                    xy=(best_loss_epoch, best_loss_val),
                    xytext=(best_loss_epoch + 5, best_loss_val + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="black", lw=1, alpha=0.8))

        ax.set_title(f'{loss_name} Evolution', fontsize=20, fontweight='semibold', pad=20)
        ax.set_xlabel('Epoch', fontsize=16, fontweight='semibold')
        ax.set_ylabel(f'{loss_name} Value', fontsize=16, fontweight='semibold')
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # 2. Mean Absolute Error
        ax = axes[0, 1]
        ax.plot(agg_data['epoch'], agg_data['train_mae_mean'], color='steelblue', label='Training MAE', linewidth=2.5)
        ax.fill_between(agg_data['epoch'], agg_data['train_mae_mean'] - agg_data['train_mae_std'], agg_data['train_mae_mean'] + agg_data['train_mae_std'], color='steelblue', alpha=0.2)
        ax.plot(agg_data['epoch'], agg_data['val_mae_mean'], color='darkorange', label='Validation MAE', linewidth=2.5, linestyle='--')
        ax.fill_between(agg_data['epoch'], agg_data['val_mae_mean'] - agg_data['val_mae_std'], agg_data['val_mae_mean'] + agg_data['val_mae_std'], color='darkorange', alpha=0.2)
        ax.set_title('Mean Absolute Error', fontsize=20, fontweight='semibold', pad=20)
        ax.set_xlabel('Epoch', fontsize=16, fontweight='semibold')
        ax.set_ylabel('MAE (kg/m²)', fontsize=16, fontweight='semibold')
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # 3. R² Score
        ax = axes[1, 0]
        ax.plot(agg_data['epoch'], agg_data['train_r2_mean'], color='steelblue', label='Training R²', linewidth=2.5)
        ax.fill_between(agg_data['epoch'], agg_data['train_r2_mean'] - agg_data['train_r2_std'], agg_data['train_r2_mean'] + agg_data['train_r2_std'], color='steelblue', alpha=0.2)
        ax.plot(agg_data['epoch'], agg_data['val_r2_mean'], color='darkorange', label='Validation R²', linewidth=2.5, linestyle='--')
        ax.fill_between(agg_data['epoch'], agg_data['val_r2_mean'] - agg_data['val_r2_std'], agg_data['val_r2_mean'] + agg_data['val_r2_std'], color='darkorange', alpha=0.2)
        ax.set_title('R² Score', fontsize=20, fontweight='semibold', pad=20)
        ax.set_xlabel('Epoch', fontsize=16, fontweight='semibold')
        ax.set_ylabel('R² Score', fontsize=16, fontweight='semibold')
        ax.set_ylim(bottom=max(agg_data['val_r2_mean'].min() - 0.1, -0.1), top=1.0)
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # 4. Validation Pearson Correlation
        ax = axes[1, 1]
        ax.plot(agg_data['epoch'], agg_data['val_pearson_mean'], color='green', label='Validation Pearson r', linewidth=2.5)
        ax.fill_between(agg_data['epoch'], agg_data['val_pearson_mean'] - agg_data['val_pearson_std'], agg_data['val_pearson_mean'] + agg_data['val_pearson_std'], color='green', alpha=0.2)
        
        # Annotation for best validation pearson
        ax.plot(best_pearson_epoch, best_pearson_val, 'o', color='red', markersize=10, label=f'Best Pearson r: {best_pearson_val:.4f}')
        ax.annotate(f'Best: {best_pearson_val:.4f}\n@ Epoch {best_pearson_epoch}',
                    xy=(best_pearson_epoch, best_pearson_val),
                    xytext=(best_pearson_epoch + 5, best_pearson_val - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="black", lw=1, alpha=0.8))

        ax.set_title('Pearson Correlation Coefficient', fontsize=20, fontweight='semibold', pad=20)
        ax.set_xlabel('Epoch', fontsize=16, fontweight='semibold')
        ax.set_ylabel('Pearson r', fontsize=16, fontweight='semibold')
        ax.set_ylim(bottom=max(agg_data['val_pearson_mean'].min() - 0.1, -0.1), top=1.0)
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        curves_path = os.path.join(self.plots_dir, 'learning_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Enhanced learning curves saved to: {curves_path}")

    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive performance metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)

        # Pearson correlation
        corr, _ = pearsonr(actual, predicted)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': corr
        }

    def plot_predictions_scatter(self):
        """Generate aggregated predicted vs actual scatter plot from validation_predictions.csv"""
        if self.data['validation_predictions'] is None:
            print("❌ Cannot plot predictions scatter: validation_predictions.csv not found")
            return

        print("📊 Generating aggregated predictions scatter plot...")

        df = self.data['validation_predictions']

        # Set style with enhanced readability
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.7)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams["font.family"] = "Arial"   

        # Create single large scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))

        # Aggregate all validation predictions across folds
        actual = df['actual_asmi'].values
        predicted = df['predicted_asmi'].values

        # Main scatter plot - all validation data points
        ax.scatter(actual, predicted, alpha=0.6, s=80, edgecolor='white',
                  linewidth=1, color='steelblue', label='Validation Predictions')

        # Perfect prediction line (y = x)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        margin = (max_val - min_val) * 0.05
        min_plot = min_val - margin
        max_plot = max_val + margin

        ax.plot([min_plot, max_plot], [min_plot, max_plot], 'r--',
               lw=3, alpha=0.8, label='Perfect Prediction (y=x)')

        # Overall regression trend line
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        X = actual.reshape(-1, 1)
        reg.fit(X, predicted)
        trend_line = reg.predict([[min_plot], [max_plot]])
        ax.plot([min_plot, max_plot], trend_line, 'b-',
               lw=2.5, alpha=0.8, label='Regression Trend')

        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(actual, predicted)

        # Performance metrics text box
        textstr = f"""Performance Metrics (All validation):
Pearson r = {metrics['pearson_r']:.3f}
R² = {metrics['r2']:.3f}
MAE = {metrics['mae']:.3f} kg/m²
RMSE = {metrics['rmse']:.3f} kg/m²
n = {len(actual)} samples"""

        props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=15,
               bbox=props, verticalalignment='top', fontfamily='Arial')

        # Formatting with enhanced typography
        ax.set_xlim(min_plot, max_plot)
        ax.set_ylim(min_plot, max_plot)
        ax.set_xlabel('Actual ASMI (kg/m²)', fontsize=18, fontweight='semibold')
        ax.set_ylabel('Predicted ASMI (kg/m²)', fontsize=18, fontweight='semibold')
        ax.set_title('Aggregated Predicted vs. Actual ASMI\n(5-Fold Cross-Validation Results)',
                    fontsize=22, fontweight='semibold', pad=25)
        ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        scatter_path = os.path.join(self.plots_dir, 'scatter_plot.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Aggregated scatter plot saved to: {scatter_path}")
        print(f"       Overall Pearson r = {metrics['pearson_r']:.3f}, R² = {metrics['r2']:.3f}")

    def plot_bland_altman(self):
        """Generate Bland-Altman plot for agreement analysis"""
        if self.data['validation_predictions'] is None:
            print("❌ Cannot plot Bland-Altman: validation_predictions.csv not found")
            return

        print("📊 Generating Bland-Altman agreement analysis plot...")

        df = self.data['validation_predictions']
        actual = df['actual_asmi'].values
        predicted = df['predicted_asmi'].values

        # Calculate Bland-Altman variables
        mean_values = (actual + predicted) / 2  # X-axis: average of two measurements
        differences = predicted - actual        # Y-axis: difference (bias)

        # Calculate statistics
        mean_diff = np.mean(differences)        # Bias
        std_diff = np.std(differences, ddof=1)  # Standard deviation

        # 95% limits of agreement
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        # Set style with enhanced readability
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.7)
        plt.rcParams['figure.facecolor'] = 'white'

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Scatter plot
        ax.scatter(mean_values, differences, alpha=0.6, s=80, edgecolor='white',
                  linewidth=1, color='steelblue', label='Validation Samples')

        # Mean difference line (bias)
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2.5,
                  label=f'Mean Bias = {mean_diff:.3f} kg/m²')

        # 95% limits of agreement
        ax.axhline(upper_loa, color='red', linestyle='--', linewidth=2,
                  label=f'95% LoA = ±{1.96 * std_diff:.3f} kg/m²')
        ax.axhline(lower_loa, color='red', linestyle='--', linewidth=2)

        # Zero line (perfect agreement)
        ax.axhline(0, color='black', linestyle=':', alpha=0.5, linewidth=1)

        # Statistics text box
        textstr = f"""Bland-Altman Analysis:
Mean Bias = {mean_diff:.3f} kg/m²
SD of Differences = {std_diff:.3f} kg/m²
95% LoA = [{lower_loa:.3f}, {upper_loa:.3f}] kg/m²
Range of LoA = {upper_loa - lower_loa:.3f} kg/m²
n = {len(actual)} samples"""

        props = dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=15,
               bbox=props, verticalalignment='top', fontfamily='Arial')

        # Formatting with enhanced typography
        ax.set_xlabel('Average of Actual and Predicted ASMI (kg/m²)', fontsize=18, fontweight='semibold')
        ax.set_ylabel('Predicted - Actual ASMI (kg/m²)', fontsize=18, fontweight='semibold')
        ax.set_title('Bland-Altman Plot: Agreement Analysis\n(AI Prediction vs. Ground Truth)',
                    fontsize=22, fontweight='semibold', pad=25)
        ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        bland_altman_path = os.path.join(self.plots_dir, 'bland_altman.png')
        plt.savefig(bland_altman_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Bland-Altman plot saved to: {bland_altman_path}")
        print(f"       Mean bias = {mean_diff:.3f} kg/m², 95% LoA = [{lower_loa:.3f}, {upper_loa:.3f}] kg/m²")

    def analyze_sarcopenia_classification(self):
        """
        Convert regression results to sarcopenia classification and analyze performance
        Classification thresholds: Male (Gender=0) ASMI < 7.0, Female (Gender=1) ASMI < 5.5
        """
        print("\n🔬 Performing sarcopenia classification analysis...")

        # Check if required data is available
        if self.data['validation_predictions'] is None or self.data['patient_data'] is None:
            print("❌ Required data not available for classification analysis")
            return

        # Merge prediction and patient data
        pred_data = self.data['validation_predictions'].copy()
        patient_data = self.data['patient_data'].copy()

        # Merge on UID
        merged_data = pred_data.merge(patient_data[['UID', 'Gender']], on='UID', how='left')

        if merged_data['Gender'].isna().any():
            print("⚠️ Warning: Some samples missing gender information")
            merged_data = merged_data.dropna(subset=['Gender'])

        print(f"✅ Merged data: {len(merged_data)} samples")
        print(f"   - Males (Gender=0): {sum(merged_data['Gender'] == 0)}")
        print(f"   - Females (Gender=1): {sum(merged_data['Gender'] == 1)}")

        # Apply sarcopenia classification thresholds
        def classify_sarcopenia(row):
            asmi = row['actual_asmi'] if 'actual_asmi' in row else row['ASMI']
            gender = row['Gender']

            if gender == 0:  # Male
                return 1 if asmi < 7.0 else 0
            else:  # Female (gender == 1)
                return 1 if asmi < 5.5 else 0

        # Apply classification to both actual and predicted ASMI
        merged_data['actual_sarcopenia'] = merged_data.apply(
            lambda row: classify_sarcopenia({'actual_asmi': row['actual_asmi'], 'Gender': row['Gender']}), axis=1
        )
        merged_data['predicted_sarcopenia'] = merged_data.apply(
            lambda row: classify_sarcopenia({'actual_asmi': row['predicted_asmi'], 'Gender': row['Gender']}), axis=1
        )

        # Calculate classification metrics
        y_true = merged_data['actual_sarcopenia']
        y_pred = merged_data['predicted_sarcopenia']
        y_pred_proba = merged_data['predicted_asmi']  # Use continuous ASMI for ROC curve

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        # For ROC curve, we need to convert ASMI scores to probabilities
        # Higher ASMI = lower probability of sarcopenia
        def asmi_to_sarcopenia_prob(asmi, gender):
            threshold = 7.0 if gender == 0 else 5.5
            # Convert to probability (sigmoid-like transformation around threshold)
            return 1 / (1 + np.exp(2 * (asmi - threshold)))

        y_pred_prob = [asmi_to_sarcopenia_prob(asmi, gender)
                      for asmi, gender in zip(merged_data['predicted_asmi'], merged_data['Gender'])]

        auc = roc_auc_score(y_true, y_pred_prob)

        # Generate classification report
        class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Sarcopenia'], output_dict=True)

        # Create visualization figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Sarcopenia'],
                   yticklabels=['Normal', 'Sarcopenia'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        # 3. Classification by Gender
        gender_results = merged_data.groupby('Gender').agg({
            'actual_sarcopenia': ['count', 'sum'],
            'predicted_sarcopenia': 'sum'
        }).round(3)

        gender_labels = ['Male', 'Female']
        actual_counts = [gender_results.loc[0, ('actual_sarcopenia', 'sum')],
                        gender_results.loc[1, ('actual_sarcopenia', 'sum')]]
        pred_counts = [gender_results.loc[0, ('predicted_sarcopenia', 'sum')],
                      gender_results.loc[1, ('predicted_sarcopenia', 'sum')]]

        x = np.arange(len(gender_labels))
        width = 0.35

        ax3.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
        ax3.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        ax3.set_xlabel('Gender')
        ax3.set_ylabel('Number of Sarcopenia Cases')
        ax3.set_title('Sarcopenia Cases by Gender')
        ax3.set_xticks(x)
        ax3.set_xticklabels(gender_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Classification Performance Metrics
        metrics_data = {
            'Metric': ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'F1-Score', 'AUC'],
            'Value': [
                accuracy,
                class_report['Sarcopenia']['recall'],
                class_report['Normal']['recall'],
                class_report['Sarcopenia']['f1-score'],
                auc
            ]
        }

        ax4.barh(metrics_data['Metric'], metrics_data['Value'], color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange'])
        ax4.set_xlabel('Score')
        ax4.set_title('Classification Performance Metrics')
        ax4.set_xlim([0, 1])

        # Add value labels on bars
        for i, v in enumerate(metrics_data['Value']):
            ax4.text(v + 0.01, i, f'{v:.3f}', va='center')

        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        classification_path = os.path.join(self.plots_dir, 'sarcopenia_classification_analysis.png')
        plt.savefig(classification_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print summary
        print(f"   ✅ Sarcopenia classification analysis saved to: {classification_path}")
        print(f"   📊 Classification Performance:")
        print(f"       - Accuracy: {accuracy:.3f}")
        print(f"       - Sensitivity (Recall): {class_report['Sarcopenia']['recall']:.3f}")
        print(f"       - Specificity: {class_report['Normal']['recall']:.3f}")
        print(f"       - F1-Score: {class_report['Sarcopenia']['f1-score']:.3f}")
        print(f"       - AUC: {auc:.3f}")
        print(f"   📈 Clinical Interpretation:")

        total_actual_sarcopenia = sum(y_true)
        total_pred_sarcopenia = sum(y_pred)
        print(f"       - Actual sarcopenia cases: {total_actual_sarcopenia}/{len(y_true)} ({100*total_actual_sarcopenia/len(y_true):.1f}%)")
        print(f"       - Predicted sarcopenia cases: {total_pred_sarcopenia}/{len(y_pred)} ({100*total_pred_sarcopenia/len(y_pred):.1f}%)")

        return {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': class_report,
            'confusion_matrix': cm
        }

    def generate_heatmaps(self, target_uids=None):
        """
        Load best model and validation data, generate and save CAM heatmaps

        Args:
            target_uids: List of specific UIDs to generate heatmaps for.
                        If None, randomly selects 4 samples.
        """
        print("\n🔥 Starting CAM heatmap generation...")

        # --- 1. 檢查必要的檔案和數據 ---
        model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        config_path = os.path.join(self.log_dir, 'configuration.txt')

        # 檢查 CSV 數據是否已載入且不為空
        if self.data.get('validation_predictions') is None or self.data['validation_predictions'].empty or \
           self.data.get('patient_data') is None or self.data['patient_data'].empty:
            print("❌ 無法生成熱圖: validation_predictions.csv 或 patient_data.csv 找不到或為空。")
            return
        if not os.path.exists(model_path):
            print(f"❌ 無法生成熱圖: best_model.pth 在 {model_path} 中找不到。")
            return

        # --- 2. 載入並準備數據 ---
        # 使用 UID 作為橋樑，合併預測結果和病患的靜態特徵
        full_df = pd.merge(self.data['validation_predictions'], self.data['patient_data'], on='UID')

        # 選取樣本進行視覺化 - 支援指定 UID 或隨機選取
        if target_uids:
            # 使用指定的 UIDs
            valid_uids, invalid_uids = self._validate_uids(target_uids, full_df)
            if invalid_uids:
                print(f"⚠️ 找不到以下 UIDs: {invalid_uids}")
            if not valid_uids:
                print("❌ 沒有有效的 UID 可供視覺化。")
                return
            sample_df = full_df[full_df['UID'].isin(valid_uids)]
            num_samples = len(sample_df)
            print(f"✅ Selected {num_samples} specified samples for visualization: {sample_df['UID'].tolist()}")
        else:
            # 隨機選取 4 個樣本
            num_samples = min(4, len(full_df))
            if num_samples == 0:
                print("❌ 沒有可供視覺化的樣本。")
                return
            sample_df = full_df.sample(n=num_samples, random_state=42)
            print(f"✅ 已隨機選取 {num_samples} 個樣本進行視覺化: {sample_df['UID'].tolist()}")

        # --- 3. 載入模型和設定 ---
        # 強制使用 CPU 避免 MPS 記憶體問題
        device = torch.device('cpu')
        print(f"🚀 將在裝置 '{device}' 上生成熱圖 (使用 CPU 避免記憶體問題)。")
        
        config_args = argparse.Namespace(config_file=config_path, train=False)
        config = Configurable(config_args, [])

        model = MODELS[config.model](
            backbone=config.backbone, n_channels=config.n_channels, config=config, use_pretrained=config.use_pretrained
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 重新啟用 CAM 增強以獲得完整的模型分析
        model.cam_enhancement = True
        print(f"✅ 模型已從 {model_path} 載入 (CAM 增強已啟用)。")

        # --- 4. Prepare dual CAM heatmap generation ---
        # 初始化兩個不同的 CAM 提取器
        # Overall Model CAM: 基於整個模型的梯度CAM
        # 關鍵：我們要分析backbone.layer4，但是梯度來自完整模型的loss
        # 這樣可以看到完整模型（包括text fusion）如何影響視覺特徵的重要性
        overall_cam_extractor = SmoothGradCAMpp(model.backbone, 'layer4')

        # Enable gradients for CAM computation and set model to eval mode but with requires_grad=True
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)

        # Re-freeze CAM generator parameters (they were unfrozen by the above loop)
        if hasattr(model, 'cam_generator'):
            for param in model.cam_generator.parameters():
                param.requires_grad_(False)

        # --- 5. 創建畫布 - 生成 2 行熱圖：Overall Model 和 CAM Generator ---
        fig, axs = plt.subplots(2, num_samples, figsize=(6 * num_samples, 12))

        # 如果只有一個樣本，axs 需要被轉換成 2D array
        if num_samples == 1:
            axs = axs.reshape(2, 1)

        for i, (_, row) in enumerate(sample_df.iterrows()):
            uid = row['UID']
            img_path = self._resolve_image_path(row['Img_path'])

            try:
                # Check if image path exists first
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                # 準備模型輸入 - 使用較小的解析度
                raw_image_np = self._load_dicom_image_small(img_path)  # 使用縮小版本
                image_tensor = transform_test(raw_image_np).unsqueeze(0).to(device)
                image_tensor.requires_grad_(True)  # 確保梯度計算用於 CAM
                clinical_data = row[TEXT_COLS].values.astype(np.float32)
                text_tensor = torch.from_numpy(clinical_data).unsqueeze(0).unsqueeze(0).to(device)

                # --- 生成兩種梯度導向的熱圖：Overall Model 和 CAM Generator ---
                # Reset model gradient state for each sample
                model.zero_grad()
                # Clear any cached gradients
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                with torch.enable_grad():
                    # === 1. Overall Model Heatmap (手動實現梯度CAM) ===
                    # 前向傳播並記錄backbone.layer4的特徵
                    activations = {}
                    def hook_layer4(module, input, output):
                        # Only retain grad if output actually requires grad
                        if output.requires_grad:
                            output.retain_grad()  # 保留非葉子節點的梯度
                            activations['layer4'] = output
                        else:
                            # If output doesn't require grad, store it anyway but warn
                            activations['layer4'] = output
                            print(f"Warning: layer4 output doesn't require grad for UID {uid}")

                    # 註冊hook到backbone.layer4
                    hook_handle = model.backbone.layer4.register_forward_hook(hook_layer4)

                    # 前向傳播完整模型（保持模型的 CAM enhancement 設定）
                    text_included = config.use_text_features
                    model_output = model(image_tensor, text_tensor, text_included=text_included)
                    regression_score = model_output[0]  # 獲取回歸輸出

                    # 計算梯度
                    model.zero_grad()
                    regression_score.backward(retain_graph=True)

                    # 獲取layer4的激活和梯度
                    layer4_activations = activations['layer4']  # (1, 512, 7, 7)
                    layer4_gradients = layer4_activations.grad  # 獲取梯度

                    # 移除hook
                    hook_handle.remove()

                    if layer4_gradients is not None:
                        # 計算權重：對每個通道的梯度進行全局平均池化
                        weights = torch.mean(layer4_gradients, dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)

                        # 計算CAM：權重加權的激活圖
                        overall_cam = torch.sum(weights * layer4_activations, dim=1, keepdim=True)  # (1, 1, 7, 7)
                        overall_cam = torch.relu(overall_cam)  # 只保留正值
                        overall_cam = overall_cam.squeeze()  # (7, 7)
                    else:
                        print("Warning: No gradients found for layer4")
                        overall_cam = torch.zeros((7, 7))

                    # 轉換為 numpy 並調整尺寸
                    h, w = raw_image_np.shape[:2]
                    overall_cam_np = overall_cam.squeeze().detach().cpu().numpy()

                    # Debug: 檢查CAM數值
                    print(f"Overall CAM shape: {overall_cam_np.shape}, min: {overall_cam_np.min():.6f}, max: {overall_cam_np.max():.6f}, mean: {overall_cam_np.mean():.6f}")

                    overall_cam_resized = cv2.resize(overall_cam_np, (w, h))

                    # 正規化到 0-1
                    if overall_cam_resized.max() > overall_cam_resized.min():
                        overall_cam_resized = (overall_cam_resized - overall_cam_resized.min()) / (overall_cam_resized.max() - overall_cam_resized.min())
                    else:
                        print(f"Warning: Overall CAM has no variation (all values same: {overall_cam_resized.min():.6f})")
                        overall_cam_resized = np.zeros_like(overall_cam_resized)

                    overall_overlay = self._overlay_heatmap(raw_image_np, overall_cam_resized)

                    # === 2. CAM Generator 獨立特徵 ===
                    # 確保 CAM generator 在正確的裝置上
                    model.cam_generator.to(device)

                    # 固定隨機種子確保 SmoothGradCAMpp 的可重現性
                    torch.manual_seed(42)
                    np.random.seed(42)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(42)

                    # SmoothGradCAMpp 需要 gradient 來註冊 hook，所以使用 enable_grad
                    with torch.enable_grad():
                        # 確保 image_tensor 需要 gradient
                        image_tensor.requires_grad_(True)

                        # 使用 CAM generator 生成 scores (獨立分析)
                        generator_scores = model.cam_generator(image_tensor)

                        # 使用 SmoothGradCAMpp 提取 CAM generator 的 activation map
                        generator_activation_map = model.cam_extractor(class_idx=0, scores=generator_scores)
                        if len(generator_activation_map) == 0:
                            raise RuntimeError("CAM Generator extractor returned empty list")
                        generator_cam = generator_activation_map[0]  # CAM generator的獨立特徵

                        # 清理 gradients 節省記憶體
                        image_tensor.requires_grad_(False)

                    # 轉換為 numpy 並調整尺寸
                    generator_cam_np = generator_cam.squeeze().detach().cpu().numpy()
                    generator_cam_resized = cv2.resize(generator_cam_np, (w, h))

                    # 正規化到 0-1
                    generator_cam_resized = (generator_cam_resized - generator_cam_resized.min()) / (generator_cam_resized.max() - generator_cam_resized.min() + 1e-8)

                # 顯示 Overall Model 梯度CAM (第一行)
                axs[0, i].imshow(overall_overlay)
                axs[0, i].set_title(f"UID: {uid}\nOverall Model (Gradient CAM)", fontsize=16, fontweight='semibold')
                axs[0, i].axis('off')

                # 顯示 CAM Generator 獨立特徵 (第二行，不疊圖)
                axs[1, i].imshow(generator_cam_resized, cmap='jet')
                axs[1, i].set_title(f"UID: {uid}\nCAM Generator (ResNet18)", fontsize=16, fontweight='semibold')
                axs[1, i].axis('off')

                print(f"✅ Generated heatmap for UID: {uid}")

            except Exception as e:
                import traceback
                import sys
                print(f"⚠️ Failed to generate heatmap for UID: {uid}: {e}")
                print(f"Full traceback:")
                traceback.print_exc(file=sys.stdout)
                # 在兩個子圖中顯示錯誤信息
                for row in range(2):
                    axs[row, i].text(0.5, 0.5, f"UID: {uid}\n[X] Heatmap generation failed\n{str(e)[:50]}...",
                                   transform=axs[row, i].transAxes, fontsize=14, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
                    axs[row, i].set_title(f"UID: {uid} (Error)", fontsize=16, fontweight='semibold', color='red')
                    axs[row, i].axis('off')

        # --- 6. 儲存圖檔 ---
        fig.suptitle('CAM Heatmap Comparison: Overall Model vs CAM Generator', fontsize=24, fontweight='semibold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Generate descriptive filename based on target UIDs
        if target_uids and len(target_uids) <= 5:  # Don't include too many UIDs in filename
            uid_suffix = "_" + "_".join(target_uids[:5])
            save_path = os.path.join(self.plots_dir, f'cam_heatmaps{uid_suffix}.png')
        else:
            save_path = os.path.join(self.plots_dir, 'cam_heatmaps.png')

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ CAM heatmap comparison (Overall Model vs CAM Generator) saved to: {save_path}")

        plt.close(fig)

    # --- Helper methods for heatmap generation ---
    def _validate_uids(self, target_uids, full_df):
        """Validate UIDs and return valid and invalid lists"""
        available_uids = set(full_df['UID'].unique())
        target_uids_set = set(target_uids)

        valid_uids = list(target_uids_set.intersection(available_uids))
        invalid_uids = list(target_uids_set - available_uids)

        return valid_uids, invalid_uids

    def _resolve_image_path(self, img_path):
        """Resolve image path properly from patient data"""
        # First check if path is already absolute
        if os.path.isabs(img_path):
            return img_path

        # Check if path exists relative to current directory
        if os.path.exists(img_path):
            return os.path.abspath(img_path)

        # Try with data directory prefix
        data_dir = os.path.join(os.path.dirname(self.log_dir), '..', '..')
        full_path = os.path.join(data_dir, img_path)
        if os.path.exists(full_path):
            return os.path.abspath(full_path)

        # Try with common prefixes
        for prefix in ['../', '../../', '../../../']:
            test_path = os.path.join(prefix, img_path)
            if os.path.exists(test_path):
                return os.path.abspath(test_path)

        # If all fails, return as-is and let error handling deal with it
        return img_path

    def _load_dicom_image_small(self, path, target_size=224):
        """從 DICOM 檔案路徑載入並預處理影像 (縮小版本節省記憶體)。"""
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # 進行視窗化以增強對比度
        if 'WindowCenter' in dcm and 'WindowWidth' in dcm:
            # Handle MultiValue (some DICOM files have multiple window settings)
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            if isinstance(center, pydicom.multival.MultiValue):
                center = float(center[0])
            else:
                center = float(center)
            if isinstance(width, pydicom.multival.MultiValue):
                width = float(width[0])
            else:
                width = float(width)
            low = center - width / 2
            high = center + width / 2
            img = np.clip(img, low, high)

        # 正規化到 0-1
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # 縮小到目標尺寸以節省記憶體
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(img, (target_size, target_size))

        # 轉為 RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return (img * 255).astype(np.uint8)

    def _load_dicom_image(self, path):
        """從 DICOM 檔案路徑載入並預處理影像。"""
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        # 進行視窗化以增強對比度 (可選，但通常效果更好)
        if 'WindowCenter' in dcm and 'WindowWidth' in dcm:
            # Handle MultiValue (some DICOM files have multiple window settings)
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            if isinstance(center, pydicom.multival.MultiValue):
                center = float(center[0])
            else:
                center = float(center)
            if isinstance(width, pydicom.multival.MultiValue):
                width = float(width[0])
            else:
                width = float(width)
            low = center - width / 2
            high = center + width / 2
            img = np.clip(img, low, high)

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) # 正規化到 0-1
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return (img * 255).astype(np.uint8)

    def _overlay_heatmap(self, raw_image_np, heatmap, colormap=cv2.COLORMAP_JET):
        """將熱圖疊加到原始影像上。"""
        h, w, _ = raw_image_np.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)

        # 將原始 RGB 影像轉換為 BGR 以便 OpenCV 處理
        raw_image_bgr = cv2.cvtColor(raw_image_np, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(raw_image_bgr, 0.6, heatmap_colored, 0.4, 0)

        # 將結果轉回 RGB 以便 Matplotlib 顯示
        return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


    def visualize_implant_detection(self, target_uids=None, threshold=240):
        """
        可視化植入物檢測結果，展示原始影像、植入物遮罩和清理後的影像

        Args:
            target_uids: List of specific UIDs to analyze. If None, randomly selects 4 samples.
            threshold: Brightness threshold for implant detection
        """
        print(f"\n🔍 開始植入物檢測分析 (閾值: {threshold})...")

        # Import ImplantDetector
        try:
            from sarcopenia_data.ImplantDetector import ImplantDetector
        except ImportError:
            print("❌ ImplantDetector 不可用，無法進行植入物檢測分析")
            return

        # Check data availability
        if self.data.get('validation_predictions') is None or self.data['validation_predictions'].empty or \
           self.data.get('patient_data') is None or self.data['patient_data'].empty:
            print("❌ 無法進行植入物檢測分析: validation_predictions.csv 或 patient_data.csv 找不到或為空。")
            return

        # Prepare data
        full_df = pd.merge(self.data['validation_predictions'], self.data['patient_data'], on='UID')

        # Select samples
        if target_uids:
            valid_uids, invalid_uids = self._validate_uids(target_uids, full_df)
            if invalid_uids:
                print(f"⚠️ 找不到以下 UIDs: {invalid_uids}")
            if not valid_uids:
                print("❌ 沒有有效的 UID 可供分析。")
                return
            sample_df = full_df[full_df['UID'].isin(valid_uids)]
            num_samples = len(sample_df)
            print(f"✅ Selected {num_samples} specified samples for analysis: {sample_df['UID'].tolist()}")
        else:
            # Randomly select 4 samples
            num_samples = min(4, len(full_df))
            if num_samples == 0:
                print("❌ 沒有可供分析的樣本。")
                return
            sample_df = full_df.sample(n=num_samples, random_state=42)
            print(f"✅ 已隨機選取 {num_samples} 個樣本進行分析: {sample_df['UID'].tolist()}")

        # Initialize implant detector
        detector = ImplantDetector(threshold=threshold)

        # Create visualization
        fig, axs = plt.subplots(3, num_samples, figsize=(6 * num_samples, 18))
        if num_samples == 1:
            axs = axs.reshape(3, 1)

        implant_stats = []

        for i, (_, row) in enumerate(sample_df.iterrows()):
            uid = row['UID']
            img_path = self._resolve_image_path(row['Img_path'])

            try:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                # Load original image
                raw_image_np = self._load_dicom_image_small(img_path)

                # Convert to grayscale for detection
                if len(raw_image_np.shape) == 3:
                    gray_image = cv2.cvtColor(raw_image_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = raw_image_np

                # Detect implants and get statistics
                mask, stats = detector.detect_implants(gray_image, return_stats=True)

                # Generate cleaned image
                cleaned_image, _ = detector.remove_implants(gray_image, strategy='inpaint', mask=mask)

                # Store statistics
                implant_stats.append({
                    'UID': uid,
                    'num_implants': stats['num_implants'],
                    'coverage': stats['image_coverage'],
                    'total_pixels': stats['total_implant_pixels']
                })

                # Display original image
                axs[0, i].imshow(raw_image_np, cmap='gray' if len(raw_image_np.shape) == 2 else None)
                axs[0, i].set_title(f"UID: {uid}\\n原始影像", fontsize=16, fontweight='semibold')
                axs[0, i].axis('off')

                # Display implant mask
                mask_overlay = np.zeros_like(raw_image_np)
                if len(raw_image_np.shape) == 3:
                    mask_overlay = np.stack([raw_image_np[:,:,0]] * 3, axis=-1)
                else:
                    mask_overlay = np.stack([raw_image_np] * 3, axis=-1)
                mask_overlay[mask] = [255, 0, 0]  # Red for implants

                axs[1, i].imshow(mask_overlay)
                implant_info = f"植入物檢測\\n數量: {stats['num_implants']}\\n覆蓋率: {stats['image_coverage']:.1%}"
                axs[1, i].set_title(f"UID: {uid}\\n{implant_info}", fontsize=16, fontweight='semibold')
                axs[1, i].axis('off')

                # Display cleaned image
                axs[2, i].imshow(cleaned_image, cmap='gray')
                axs[2, i].set_title(f"UID: {uid}\\n清理後影像", fontsize=16, fontweight='semibold')
                axs[2, i].axis('off')

                print(f"✅ UID: {uid} - 檢測到 {stats['num_implants']} 個植入物 (覆蓋率: {stats['image_coverage']:.1%})")

            except Exception as e:
                print(f"⚠️ 無法為 UID: {uid} 進行植入物檢測: {e}")
                # Display error message in all three rows
                for row in range(3):
                    axs[row, i].text(0.5, 0.5, f"UID: {uid}\\n❌ 分析失敗\\n{str(e)[:50]}...",
                                   transform=axs[row, i].transAxes, fontsize=14, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
                    axs[row, i].set_title(f"UID: {uid} (Error)", fontsize=16, fontweight='semibold', color='red')
                    axs[row, i].axis('off')

        # Overall title and layout
        fig.suptitle(f"植入物檢測分析 (閾值: {threshold})", fontsize=24, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot
        if target_uids and len(target_uids) <= 5:
            uid_suffix = "_" + "_".join(target_uids[:5])
            save_path = os.path.join(self.plots_dir, f'implant_detection{uid_suffix}.png')
        else:
            save_path = os.path.join(self.plots_dir, 'implant_detection.png')

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ 植入物檢測分析已儲存至: {save_path}")

        # Print summary statistics
        if implant_stats:
            total_with_implants = sum(1 for s in implant_stats if s['num_implants'] > 0)
            avg_coverage = np.mean([s['coverage'] for s in implant_stats if s['num_implants'] > 0])
            print(f"\\n📊 植入物檢測統計:")
            print(f"   - 有植入物的樣本: {total_with_implants}/{len(implant_stats)}")
            if total_with_implants > 0:
                print(f"   - 平均覆蓋率: {avg_coverage:.1%}")
            print(f"   - 使用閾值: {threshold}")

        plt.close(fig)

    def generate_all_plots(self):
        """Generate all 6 medical research plots"""
        print("🎨 Generating all medical research plots...")

        self.plot_training_curves()
        self.plot_predictions_scatter()
        self.plot_bland_altman()
        self.analyze_sarcopenia_classification()
        self.generate_heatmaps()
        self.visualize_implant_detection()

        print(f"\n🎉 All 6 plots generated successfully!")
        print(f"📁 Plots saved to: {self.plots_dir}")
        print(f"📊 Generated plots:")
        print(f"   1. Aggregated Learning Curves (training_curves.png)")
        print(f"   2. Predicted vs Actual Scatter Plot (predictions_scatter.png)")
        print(f"   3. Bland-Altman Agreement Analysis (bland_altman.png)")
        print(f"   4. Sarcopenia Classification Analysis (sarcopenia_classification_analysis.png)")
        print(f"   5. CAM Heatmap Comparison: Overall Model vs CAM Generator (cam_heatmaps.png)")
        print(f"   6. Implant Detection Analysis (implant_detection.png)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from training CSV data",
        epilog="""
Examples:
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type all
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type classification
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type heatmap
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type heatmap --UID U001 U002 U003
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type implant
  %(prog)s --log-dir ./log/data/ResNet/0_run_ASMI-Reg --type implant --UID U001 --threshold 230
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--log-dir', required=True, help='Path to training log directory')
    parser.add_argument('--type', choices=['all', 'curves', 'scatter', 'bland_altman', 'classification', 'heatmap', 'implant'],
                       default='all', help='Type of plot to generate')
    parser.add_argument('--UID', nargs='*', help='Specific UIDs to generate heatmaps/implant analysis for (space-separated). Only works with --type heatmap or implant')
    parser.add_argument('--threshold', type=int, default=240, help='Brightness threshold for implant detection (200-250). Only works with --type implant')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"❌ Log directory not found: {args.log_dir}")
        return

    # Warn if UID is provided for non-heatmap/implant types
    if args.UID and args.type not in ['heatmap', 'implant']:
        print(f"⚠️  Warning: --UID parameter is only used with --type heatmap or implant. Ignoring UIDs for type '{args.type}'")

    # Warn if threshold is provided for non-implant types
    if args.threshold != 240 and args.type != 'implant':
        print(f"⚠️  Warning: --threshold parameter is only used with --type implant. Ignoring threshold for type '{args.type}'")

    analyzer = TrainingAnalyzer(args.log_dir)

    if args.type == 'all':
        analyzer.generate_all_plots()
    elif args.type == 'curves':
        analyzer.plot_training_curves()
    elif args.type == 'scatter':
        analyzer.plot_predictions_scatter()
    elif args.type == 'bland_altman':
        analyzer.plot_bland_altman()
    elif args.type == 'classification':
        analyzer.analyze_sarcopenia_classification()
    elif args.type == 'heatmap':
        analyzer.generate_heatmaps(target_uids=args.UID)
    elif args.type == 'implant':
        analyzer.visualize_implant_detection(target_uids=args.UID, threshold=args.threshold)


if __name__ == '__main__':
    main()