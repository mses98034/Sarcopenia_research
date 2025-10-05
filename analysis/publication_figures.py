#!/usr/bin/env python
"""
Publication-Ready Figures Generator for ASMI Regression Analysis
Generates comparative figures between cross-validation and external test set results

Usage:
    python publication_figures.py --train-log LOG_DIR --test-log TEST_DIR --output OUTPUT_DIR
"""

import sys
sys.path.extend(["../", "./"])

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    confusion_matrix, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import torch
from PIL import Image
import pydicom
import cv2
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from driver.Config import Configurable
from models import MODELS
from driver import transform_test
from commons.constant import TEXT_COLS, UID, PATH, ASMI

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


class PublicationFigureGenerator:
    """
    Generate publication-ready figures comparing cross-validation and external test results
    """

    def __init__(self, train_log_dir, test_log_dir, output_dir='../results/publication_figures'):
        """
        Initialize figure generator

        Args:
            train_log_dir: Path to training log directory (cross-validation results)
            test_log_dir: Path to test log directory (external test results)
            output_dir: Output directory for generated figures
        """
        self.train_log_dir = train_log_dir
        self.test_log_dir = test_log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("PUBLICATION FIGURE GENERATOR")
        print(f"{'='*60}")
        print(f"Training log: {train_log_dir}")
        print(f"Test log: {test_log_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

        # Load data
        self.train_data = self._load_training_data()
        self.test_data = self._load_test_data()

    def _load_training_data(self):
        """Load and merge training (cross-validation) data"""
        print("📂 Loading training data...")

        csv_dir = os.path.join(self.train_log_dir, 'csv_data')

        # Load predictions and patient data
        val_pred_path = os.path.join(csv_dir, 'validation_predictions.csv')
        patient_path = os.path.join(csv_dir, 'patient_data.csv')

        if not os.path.exists(val_pred_path):
            raise FileNotFoundError(f"Validation predictions not found: {val_pred_path}")
        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"Patient data not found: {patient_path}")

        val_pred = pd.read_csv(val_pred_path)
        patient_data = pd.read_csv(patient_path)

        # Clean column names
        val_pred.columns = [col.strip() for col in val_pred.columns]
        patient_data.columns = [col.strip() for col in patient_data.columns]

        # Merge on UID to get complete information
        merged = pd.merge(val_pred, patient_data, on='UID', how='left')

        print(f"   ✅ Training data loaded: {len(merged)} samples from {val_pred['fold'].nunique()} folds")
        print(f"      Columns: {list(merged.columns)}")

        return merged

    def _load_test_data(self):
        """Load and merge test (external) data"""
        print("📂 Loading test data...")

        csv_dir = os.path.join(self.test_log_dir, 'csv_data')

        # Load predictions and patient data
        test_pred_path = os.path.join(csv_dir, 'test_predictions.csv')
        test_patient_path = os.path.join(csv_dir, 'test_patient_data.csv')

        if not os.path.exists(test_pred_path):
            raise FileNotFoundError(f"Test predictions not found: {test_pred_path}")
        if not os.path.exists(test_patient_path):
            raise FileNotFoundError(f"Test patient data not found: {test_patient_path}")

        test_pred = pd.read_csv(test_pred_path)
        test_patient = pd.read_csv(test_patient_path)

        # Clean column names
        test_pred.columns = [col.strip() for col in test_pred.columns]
        test_patient.columns = [col.strip() for col in test_patient.columns]

        # Merge on UID
        merged = pd.merge(test_pred, test_patient, on='UID', how='left')

        print(f"   ✅ Test data loaded: {len(merged)} samples")
        print(f"      Columns: {list(merged.columns)}")

        return merged

    def calculate_classification_labels(self, df):
        """
        Calculate binary classification labels for sarcopenia diagnosis

        Args:
            df: DataFrame with 'predicted_asmi', 'Low_muscle_mass', 'Gender'

        Returns:
            y_true: Ground truth from Low_muscle_mass column
            y_pred: Predicted binary labels (from predicted_asmi + gender threshold)
            y_score: Continuous score for ROC (gender-specific transformation)
        """
        # Get ground truth from Low_muscle_mass column
        if 'Low_muscle_mass' not in df.columns:
            raise ValueError(
                "Low_muscle_mass column not found in dataframe. "
                "Cannot perform classification analysis. "
                "Please ensure patient_data.csv includes Low_muscle_mass column."
            )

        y_true = df['Low_muscle_mass'].values.astype(int)

        # Get predicted ASMI and gender
        predicted_asmi = df['predicted_asmi'].values
        genders = df['Gender'].values

        # Calculate predicted binary labels from predicted ASMI using gender-specific thresholds
        y_pred = np.zeros(len(predicted_asmi), dtype=int)
        male_mask = (genders == 0)
        female_mask = (genders == 1)
        y_pred[male_mask] = (predicted_asmi[male_mask] < 7.0).astype(int)
        y_pred[female_mask] = (predicted_asmi[female_mask] < 5.5).astype(int)

        # For ROC curve: use standardized distance from gender-specific thresholds as risk score
        # This approach considers gender-specific diagnostic criteria without arbitrary transformations
        def calculate_risk_score(asmi, gender):
            threshold = 7.0 if gender == 0 else 5.5
            # Distance from threshold: positive score = below threshold (sarcopenia risk)
            # Higher positive score = further below threshold = higher risk
            return threshold - asmi

        y_score = np.array([
            calculate_risk_score(asmi, gender)
            for asmi, gender in zip(predicted_asmi, genders)
        ])

        return y_true, y_pred, y_score

    def calculate_classification_metrics(self, y_true, y_pred, y_score):
        """Calculate comprehensive classification metrics"""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_true, y_score),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

        return metrics

    def figure1_scatter_comparison(self):
        """
        Figure 1: Scatter plot comparison (Cross-validation vs External test)
        """
        print("\n📊 Generating Figure 1: Scatter Plot Comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left: Cross-validation scatter ---
        ax = axes[0]
        train_actual = self.train_data['actual_asmi']
        train_pred = self.train_data['predicted_asmi']

        # Calculate metrics
        train_r, train_p = pearsonr(train_actual, train_pred)
        train_r2 = r2_score(train_actual, train_pred)
        train_mae = mean_absolute_error(train_actual, train_pred)

        # Scatter plot
        ax.scatter(train_actual, train_pred, alpha=0.5, s=30, c='#2E86AB', edgecolors='none')

        # Identity line
        lims = [
            min(train_actual.min(), train_pred.min()),
            max(train_actual.max(), train_pred.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1.5, label='Perfect prediction')

        # Regression line
        z = np.polyfit(train_actual, train_pred, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), "r-", alpha=0.8, linewidth=2, label=f'Regression line')

        ax.set_xlabel('Actual ASMI (kg/m²)', fontweight='bold')
        ax.set_ylabel('Predicted ASMI (kg/m²)', fontweight='bold')
        ax.set_title('Cross-Validation (5-Fold)', fontweight='bold', fontsize=13)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add metrics text
        metrics_text = f'Pearson r = {train_r:.3f} (p < 0.001)\nR² = {train_r2:.3f}\nMAE = {train_mae:.3f} kg/m²\nn = {len(train_actual)}'
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        # --- Right: External test scatter ---
        ax = axes[1]
        test_actual = self.test_data['actual_asmi']
        test_pred = self.test_data['predicted_asmi']

        # Calculate metrics
        test_r, test_p = pearsonr(test_actual, test_pred)
        test_r2 = r2_score(test_actual, test_pred)
        test_mae = mean_absolute_error(test_actual, test_pred)

        # Scatter plot
        ax.scatter(test_actual, test_pred, alpha=0.5, s=30, c='#A23B72', edgecolors='none')

        # Identity line
        lims = [
            min(test_actual.min(), test_pred.min()),
            max(test_actual.max(), test_pred.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1.5, label='Perfect prediction')

        # Regression line
        z = np.polyfit(test_actual, test_pred, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), "r-", alpha=0.8, linewidth=2, label=f'Regression line')

        ax.set_xlabel('Actual ASMI (kg/m²)', fontweight='bold')
        ax.set_ylabel('Predicted ASMI (kg/m²)', fontweight='bold')
        ax.set_title('External Test Set', fontweight='bold', fontsize=13)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add metrics text
        metrics_text = f'Pearson r = {test_r:.3f} (p < 0.001)\nR² = {test_r2:.3f}\nMAE = {test_mae:.3f} kg/m²\nn = {len(test_actual)}'
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'figure1_scatter_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 1 saved: {output_path}")

    def figure2_bland_altman_comparison(self):
        """
        Figure 2: Bland-Altman plot comparison (Cross-validation vs External test)
        """
        print("\n📊 Generating Figure 2: Bland-Altman Plot Comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left: Cross-validation Bland-Altman ---
        ax = axes[0]
        train_actual = self.train_data['actual_asmi'].values
        train_pred = self.train_data['predicted_asmi'].values

        mean_vals = (train_actual + train_pred) / 2
        diff_vals = train_pred - train_actual

        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        ax.scatter(mean_vals, diff_vals, alpha=0.5, s=30, c='#2E86AB', edgecolors='none')
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean diff: {mean_diff:.3f}')
        ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
        ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Mean of Actual and Predicted ASMI (kg/m²)', fontweight='bold')
        ax.set_ylabel('Difference (Predicted - Actual) (kg/m²)', fontweight='bold')
        ax.set_title('Cross-Validation (5-Fold)', fontweight='bold', fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Right: External test Bland-Altman ---
        ax = axes[1]
        test_actual = self.test_data['actual_asmi'].values
        test_pred = self.test_data['predicted_asmi'].values

        mean_vals = (test_actual + test_pred) / 2
        diff_vals = test_pred - test_actual

        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        ax.scatter(mean_vals, diff_vals, alpha=0.5, s=30, c='#A23B72', edgecolors='none')
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean diff: {mean_diff:.3f}')
        ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
        ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Mean of Actual and Predicted ASMI (kg/m²)', fontweight='bold')
        ax.set_ylabel('Difference (Predicted - Actual) (kg/m²)', fontweight='bold')
        ax.set_title('External Test Set', fontweight='bold', fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'figure2_bland_altman_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 2 saved: {output_path}")

    def figure3_roc_comparison(self):
        """
        Figure 3: ROC curve comparison (Cross-validation vs External test)
        """
        print("\n📊 Generating Figure 3: ROC Curve Comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left: Cross-validation ROC ---
        ax = axes[0]
        train_y_true, train_y_pred, train_y_score = self.calculate_classification_labels(self.train_data)

        # Print diagnostic info
        print(f"   Training set:")
        print(f"      Total samples: {len(train_y_true)}")
        print(f"      Actual sarcopenia (ground truth): {sum(train_y_true)} ({100*sum(train_y_true)/len(train_y_true):.1f}%)")
        print(f"      Predicted sarcopenia: {sum(train_y_pred)} ({100*sum(train_y_pred)/len(train_y_pred):.1f}%)")
        print(f"      Score range: [{train_y_score.min():.3f}, {train_y_score.max():.3f}]")

        train_fpr, train_tpr, _ = roc_curve(train_y_true, train_y_score)
        train_auc = auc(train_fpr, train_tpr)

        ax.plot(train_fpr, train_tpr, color='#2E86AB', lw=2.5,
                label=f'ROC curve (AUC = {train_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Chance (AUC = 0.500)')

        # Mark threshold-based operating point on ROC curve
        # Calculate TPR and FPR when using gender-specific thresholds
        tn = np.sum((train_y_true == 0) & (train_y_pred == 0))
        fp = np.sum((train_y_true == 0) & (train_y_pred == 1))
        fn = np.sum((train_y_true == 1) & (train_y_pred == 0))
        tp = np.sum((train_y_true == 1) & (train_y_pred == 1))

        threshold_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        threshold_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax.scatter(threshold_fpr, threshold_tpr, color='red', s=150, zorder=5,
                  marker='*', edgecolors='darkred', linewidths=1.5,
                  label=f'Threshold point ({threshold_tpr:.2f}, {threshold_fpr:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('Cross-Validation (5-Fold)', fontweight='bold', fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add threshold note
        threshold_text = 'Sarcopenia criteria:\nMale: ASMI < 7.0 kg/m²\nFemale: ASMI < 5.5 kg/m²'
        ax.text(0.98, 0.02, threshold_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                fontsize=9)

        # --- Right: External test ROC ---
        ax = axes[1]
        test_y_true, test_y_pred, test_y_score = self.calculate_classification_labels(self.test_data)

        # Print diagnostic info
        print(f"   Test set:")
        print(f"      Total samples: {len(test_y_true)}")
        print(f"      Actual sarcopenia (ground truth): {sum(test_y_true)} ({100*sum(test_y_true)/len(test_y_true):.1f}%)")
        print(f"      Predicted sarcopenia: {sum(test_y_pred)} ({100*sum(test_y_pred)/len(test_y_pred):.1f}%)")
        print(f"      Score range: [{test_y_score.min():.3f}, {test_y_score.max():.3f}]")

        test_fpr, test_tpr, _ = roc_curve(test_y_true, test_y_score)
        test_auc = auc(test_fpr, test_tpr)

        ax.plot(test_fpr, test_tpr, color='#A23B72', lw=2.5,
                label=f'ROC curve (AUC = {test_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Chance (AUC = 0.500)')

        # Mark threshold-based operating point on ROC curve
        # Calculate TPR and FPR when using gender-specific thresholds
        tn = np.sum((test_y_true == 0) & (test_y_pred == 0))
        fp = np.sum((test_y_true == 0) & (test_y_pred == 1))
        fn = np.sum((test_y_true == 1) & (test_y_pred == 0))
        tp = np.sum((test_y_true == 1) & (test_y_pred == 1))

        threshold_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        threshold_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax.scatter(threshold_fpr, threshold_tpr, color='red', s=150, zorder=5,
                  marker='*', edgecolors='darkred', linewidths=1.5,
                  label=f'Threshold point ({threshold_tpr:.2f}, {threshold_fpr:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('External Test Set', fontweight='bold', fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add threshold note
        ax.text(0.98, 0.02, threshold_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'figure3_roc_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 3 saved: {output_path}")

        return train_auc, test_auc

    def figure4_metrics_table(self):
        """
        Figure 4: Classification metrics comparison table
        """
        print("\n📊 Generating Figure 4: Classification Metrics Table...")

        # Calculate classification metrics for both datasets
        train_y_true, train_y_pred, train_y_score = self.calculate_classification_labels(self.train_data)
        train_metrics = self.calculate_classification_metrics(train_y_true, train_y_pred, train_y_score)

        test_y_true, test_y_pred, test_y_score = self.calculate_classification_labels(self.test_data)
        test_metrics = self.calculate_classification_metrics(test_y_true, test_y_pred, test_y_score)

        # Create comparison dataframe
        metrics_names = ['AUC-ROC', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score', 'PPV', 'NPV']
        metrics_keys = ['auc', 'sensitivity', 'specificity', 'accuracy', 'f1_score', 'ppv', 'npv']

        comparison_data = {
            'Metric': metrics_names,
            'Cross-Validation (5-Fold)': [train_metrics[key] for key in metrics_keys],
            'External Test Set': [test_metrics[key] for key in metrics_keys]
        }

        df = pd.DataFrame(comparison_data)

        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'figure4_metrics_table.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"   ✅ Metrics table CSV saved: {csv_path}")

        # Create horizontal bar chart (2-panel layout)
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Define colors for different metric types
        metric_colors = {
            'AUC-ROC': '#2E86AB',           # Blue
            'Sensitivity': '#27AE60',       # Green
            'Specificity': '#27AE60',       # Green
            'Accuracy': '#F39C12',          # Orange
            'F1 Score': '#F39C12',          # Orange
            'PPV': '#8E44AD',               # Purple
            'NPV': '#8E44AD'                # Purple
        }

        colors = [metric_colors[metric] for metric in metrics_names]
        y_positions = np.arange(len(metrics_names))

        # Left panel: Cross-Validation
        ax_left = axes[0]
        train_values = [train_metrics[key] for key in metrics_keys]
        bars_left = ax_left.barh(y_positions, train_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for bar, value in zip(bars_left, train_values):
            ax_left.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center', ha='left', fontsize=10, fontweight='bold')

        ax_left.set_yticks(y_positions)
        ax_left.set_yticklabels(metrics_names, fontsize=11)
        ax_left.set_xlabel('Metric Value', fontsize=11, fontweight='bold')
        ax_left.set_title('Cross-Validation (5-Fold)', fontsize=13, fontweight='bold', pad=10)
        ax_left.set_xlim([0, 1.05])
        ax_left.grid(axis='x', alpha=0.3, linestyle='--')
        ax_left.invert_yaxis()  # Highest metric on top

        # Right panel: External Test Set
        ax_right = axes[1]
        test_values = [test_metrics[key] for key in metrics_keys]
        bars_right = ax_right.barh(y_positions, test_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for bar, value in zip(bars_right, test_values):
            ax_right.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                         f'{value:.3f}', va='center', ha='left', fontsize=10, fontweight='bold')

        ax_right.set_yticks(y_positions)
        ax_right.set_yticklabels(metrics_names, fontsize=11)
        ax_right.set_xlabel('Metric Value', fontsize=11, fontweight='bold')
        ax_right.set_title('External Test Set', fontsize=13, fontweight='bold', pad=10)
        ax_right.set_xlim([0, 1.05])
        ax_right.grid(axis='x', alpha=0.3, linestyle='--')
        ax_right.invert_yaxis()

        # Overall title
        fig.suptitle('Classification Performance Comparison (Sarcopenia Diagnosis)',
                    fontsize=15, fontweight='bold', y=0.98)

        # Add confusion matrix info at bottom
        cm_text = (f"Cross-Validation Confusion Matrix: TP={train_metrics['tp']}, "
                  f"TN={train_metrics['tn']}, FP={train_metrics['fp']}, FN={train_metrics['fn']}  |  "
                  f"External Test Confusion Matrix: TP={test_metrics['tp']}, "
                  f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}")

        plt.figtext(0.5, 0.02, cm_text, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        output_path = os.path.join(self.output_dir, 'figure4_metrics_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 4 saved: {output_path}")

        # Print metrics to console
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)

    def figure5_heatmap_visualization(self, target_uids=None):
        """
        Figure 5: CAM heatmap visualization (Overall Model gradient-based)
        Copied and adapted from plot.py Overall Model CAM generation logic

        Args:
            target_uids: List of UIDs to generate heatmaps for. If None, randomly selects 4 samples.
        """
        print("\n📊 Generating Figure 5: CAM Heatmap Visualization (Overall Model)...")

        # Load model and configuration
        model_path = os.path.join(self.train_log_dir, 'checkpoint', 'best_model.pth')
        config_path = os.path.join(self.train_log_dir, 'configuration.txt')

        if not os.path.exists(model_path):
            print(f"   ❌ Model not found: {model_path}")
            print("   ⚠️  Skipping heatmap generation")
            return

        if not os.path.exists(config_path):
            print(f"   ❌ Configuration not found: {config_path}")
            print("   ⚠️  Skipping heatmap generation")
            return

        # Combine train and test data to allow selecting from both
        combined_data = pd.concat([self.train_data, self.test_data], ignore_index=True)

        # Select samples
        if target_uids:
            valid_uids = [uid for uid in target_uids if uid in combined_data['UID'].values]
            if not valid_uids:
                print(f"   ❌ No valid UIDs found in train or test data")
                return
            sample_df = combined_data[combined_data['UID'].isin(valid_uids)]
            print(f"   ✅ Selected {len(sample_df)} specified samples: {valid_uids}")
        else:
            # Default: randomly select from test data
            num_samples = min(4, len(self.test_data))
            sample_df = self.test_data.sample(n=num_samples, random_state=42)
            print(f"   ✅ Randomly selected {num_samples} samples from test data: {sample_df['UID'].tolist()}")

        # Load model
        device = torch.device('cpu')
        config_args = argparse.Namespace(config_file=config_path, train=False)
        config = Configurable(config_args, [])

        model = MODELS[config.model](
            backbone=config.backbone,
            n_channels=config.n_channels,
            config=config,
            use_pretrained=config.use_pretrained
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Enable CAM enhancement and gradients
        if hasattr(model, 'cam_enhancement'):
            model.cam_enhancement = True

        for param in model.parameters():
            param.requires_grad_(True)

        print(f"   ✅ Model loaded from {model_path} (CAM enhancement enabled)")

        # Create figure
        num_samples = len(sample_df)
        fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
        if num_samples == 1:
            axes = [axes]

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            uid = row['UID']
            img_path = row['Img_path']

            # Resolve image path
            if not os.path.isabs(img_path):
                for prefix in ['../../', '../', './']:
                    test_path = os.path.join(prefix, img_path)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break

            if not os.path.exists(img_path):
                print(f"   ⚠️  Image not found for UID {uid}: {img_path}")
                axes[idx].text(0.5, 0.5, f'Image not found\n{uid}',
                             ha='center', va='center', fontsize=12)
                axes[idx].axis('off')
                continue

            try:
                # Load image using helper function (same as plot.py)
                raw_image_np = self._load_dicom_image_small(img_path)

                # Prepare model input
                image_tensor = transform_test(raw_image_np).unsqueeze(0).to(device)
                image_tensor.requires_grad_(True)

                clinical_data = row[TEXT_COLS].values.astype(np.float32)
                text_tensor = torch.from_numpy(clinical_data).unsqueeze(0).unsqueeze(0).to(device)

                # === Overall Model Gradient CAM (copied from plot.py) ===
                model.zero_grad()

                with torch.enable_grad():
                    # Register hook to capture layer4 activations
                    activations = {}
                    def hook_layer4(module, input, output):
                        if output.requires_grad:
                            output.retain_grad()
                            activations['layer4'] = output
                        else:
                            activations['layer4'] = output

                    hook_handle = model.backbone.layer4.register_forward_hook(hook_layer4)

                    # Forward pass
                    text_included = config.use_text_features
                    model_output = model(image_tensor, text_tensor, text_included=text_included)
                    regression_score = model_output[0]

                    # Backward pass
                    model.zero_grad()
                    regression_score.backward(retain_graph=True)

                    # Get activations and gradients
                    layer4_activations = activations['layer4']
                    layer4_gradients = layer4_activations.grad

                    hook_handle.remove()

                    if layer4_gradients is not None:
                        # Compute weights: global average pooling of gradients
                        weights = torch.mean(layer4_gradients, dim=(2, 3), keepdim=True)

                        # Compute CAM: weighted sum of activations
                        overall_cam = torch.sum(weights * layer4_activations, dim=1, keepdim=True)
                        overall_cam = torch.relu(overall_cam)
                        overall_cam = overall_cam.squeeze()
                    else:
                        print(f"   Warning: No gradients found for layer4 (UID: {uid})")
                        overall_cam = torch.zeros((7, 7))

                    # Convert to numpy and resize
                    h, w = raw_image_np.shape[:2]
                    overall_cam_np = overall_cam.squeeze().detach().cpu().numpy()

                    print(f"   Overall CAM - UID {uid}: shape={overall_cam_np.shape}, "
                          f"min={overall_cam_np.min():.6f}, max={overall_cam_np.max():.6f}, "
                          f"mean={overall_cam_np.mean():.6f}")

                    overall_cam_resized = cv2.resize(overall_cam_np, (w, h))

                    # Normalize to 0-1
                    if overall_cam_resized.max() > overall_cam_resized.min():
                        overall_cam_resized = (overall_cam_resized - overall_cam_resized.min()) / \
                                            (overall_cam_resized.max() - overall_cam_resized.min())
                    else:
                        print(f"   Warning: Overall CAM has no variation for UID {uid}")
                        overall_cam_resized = np.zeros_like(overall_cam_resized)

                    # Overlay heatmap using helper function
                    overall_overlay = self._overlay_heatmap(raw_image_np, overall_cam_resized)

                # Display
                axes[idx].imshow(overall_overlay)

                # Add title with metrics
                actual_asmi = row['actual_asmi'] if 'actual_asmi' in row else row.get('ASMI', 'N/A')
                pred_asmi = row['predicted_asmi']
                axes[idx].set_title(f'Actual: {actual_asmi:.2f} | Predicted: {pred_asmi:.2f} kg/m²',
                                   fontsize=12, fontweight='bold', fontfamily='sans') # UID: {uid}\n
                axes[idx].axis('off')

                print(f"   ✅ Generated heatmap for UID: {uid}")

            except Exception as e:
                import traceback
                print(f"   ⚠️  Failed to generate heatmap for UID {uid}: {e}")
                print(f"   Full traceback:")
                traceback.print_exc()

                axes[idx].text(0.5, 0.5, f'UID: {uid}\n[X] Heatmap generation failed\n{str(e)[:50]}...',
                             transform=axes[idx].transAxes, fontsize=12, ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
                axes[idx].set_title(f'UID: {uid} (Error)', fontsize=12, fontweight='semibold', color='red')
                axes[idx].axis('off')

        # plt.suptitle('CAM Heatmap Visualization (Overall Model - Gradient CAM)',
        #             fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'figure5_heatmaps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 5 saved: {output_path}")

    def figure6_subgroup_analysis(self):
        """
        Figure 6: Subgroup analysis (Gender and Age groups)
        Left column: Cross-validation | Right column: External test
        Rows: Pearson r, R², MAE, AUC
        """
        print("\n📊 Generating Figure 6: Subgroup Analysis...")

        # Define age groups (geriatric medicine standard)
        def assign_age_group(age):
            if age < 65:
                return '<65'
            elif age <= 80:
                return '65-80'
            else:
                return '>80'

        # Prepare training data
        train_df = self.train_data.copy()
        train_df['Age_Group'] = train_df['Age'].apply(assign_age_group)
        train_df['Gender_Label'] = train_df['Gender'].apply(lambda x: 'Male' if x == 0 else 'Female')

        # Prepare test data
        test_df = self.test_data.copy()
        test_df['Age_Group'] = test_df['Age'].apply(assign_age_group)
        test_df['Gender_Label'] = test_df['Gender'].apply(lambda x: 'Male' if x == 0 else 'Female')

        # Calculate metrics for each subgroup
        subgroups = []
        train_metrics = {'Pearson_r': [], 'R2': [], 'MAE': [], 'AUC': []}
        test_metrics = {'Pearson_r': [], 'R2': [], 'MAE': [], 'AUC': []}

        # Gender subgroups
        for gender in ['Male', 'Female']:
            train_subset = train_df[train_df['Gender_Label'] == gender]
            test_subset = test_df[test_df['Gender_Label'] == gender]

            if len(train_subset) > 2 and len(test_subset) > 2:
                # Training metrics
                train_r, _ = pearsonr(train_subset['actual_asmi'], train_subset['predicted_asmi'])
                train_r2 = r2_score(train_subset['actual_asmi'], train_subset['predicted_asmi'])
                train_mae = mean_absolute_error(train_subset['actual_asmi'], train_subset['predicted_asmi'])

                train_y_true, train_y_pred, train_y_score = self.calculate_classification_labels(train_subset)
                train_auc = roc_auc_score(train_y_true, train_y_score)

                # Test metrics
                test_r, _ = pearsonr(test_subset['actual_asmi'], test_subset['predicted_asmi'])
                test_r2 = r2_score(test_subset['actual_asmi'], test_subset['predicted_asmi'])
                test_mae = mean_absolute_error(test_subset['actual_asmi'], test_subset['predicted_asmi'])

                test_y_true, test_y_pred, test_y_score = self.calculate_classification_labels(test_subset)
                test_auc = roc_auc_score(test_y_true, test_y_score)

                subgroups.append(gender)
                train_metrics['Pearson_r'].append(train_r)
                train_metrics['R2'].append(train_r2)
                train_metrics['MAE'].append(train_mae)
                train_metrics['AUC'].append(train_auc)

                test_metrics['Pearson_r'].append(test_r)
                test_metrics['R2'].append(test_r2)
                test_metrics['MAE'].append(test_mae)
                test_metrics['AUC'].append(test_auc)

        # Age subgroups
        for age_group in ['<65', '65-80', '>80']:
            train_subset = train_df[train_df['Age_Group'] == age_group]
            test_subset = test_df[test_df['Age_Group'] == age_group]

            if len(train_subset) > 2 and len(test_subset) > 2:
                # Training metrics
                train_r, _ = pearsonr(train_subset['actual_asmi'], train_subset['predicted_asmi'])
                train_r2 = r2_score(train_subset['actual_asmi'], train_subset['predicted_asmi'])
                train_mae = mean_absolute_error(train_subset['actual_asmi'], train_subset['predicted_asmi'])

                train_y_true, train_y_pred, train_y_score = self.calculate_classification_labels(train_subset)
                train_auc = roc_auc_score(train_y_true, train_y_score)

                # Test metrics
                test_r, _ = pearsonr(test_subset['actual_asmi'], test_subset['predicted_asmi'])
                test_r2 = r2_score(test_subset['actual_asmi'], test_subset['predicted_asmi'])
                test_mae = mean_absolute_error(test_subset['actual_asmi'], test_subset['predicted_asmi'])

                test_y_true, test_y_pred, test_y_score = self.calculate_classification_labels(test_subset)
                test_auc = roc_auc_score(test_y_true, test_y_score)

                subgroups.append(age_group)
                train_metrics['Pearson_r'].append(train_r)
                train_metrics['R2'].append(train_r2)
                train_metrics['MAE'].append(train_mae)
                train_metrics['AUC'].append(train_auc)

                test_metrics['Pearson_r'].append(test_r)
                test_metrics['R2'].append(test_r2)
                test_metrics['MAE'].append(test_mae)
                test_metrics['AUC'].append(test_auc)

        # Save metrics as CSV
        results_df = pd.DataFrame({
            'Subgroup': subgroups,
            'Train_Pearson_r': train_metrics['Pearson_r'],
            'Test_Pearson_r': test_metrics['Pearson_r'],
            'Train_R2': train_metrics['R2'],
            'Test_R2': test_metrics['R2'],
            'Train_MAE': train_metrics['MAE'],
            'Test_MAE': test_metrics['MAE'],
            'Train_AUC': train_metrics['AUC'],
            'Test_AUC': test_metrics['AUC']
        })

        csv_path = os.path.join(self.output_dir, 'figure6_subgroup_metrics.csv')
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"   ✅ Subgroup metrics CSV saved: {csv_path}")

        # Create figure: 4 rows × 2 columns
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))

        metrics_list = [
            ('Pearson_r', 'Pearson Correlation (r)', (0.0, 1.0)),
            ('R2', 'R² Score', (0.0, 1.0)),
            ('MAE', 'MAE (kg/m²)', (0.0, None)),
            ('AUC', 'AUC-ROC', (0.0, 1.0))
        ]

        # Colors for subgroups
        colors = {
            'Male': '#2E86AB',
            'Female': '#A23B72',
            '<65': '#90EE90',
            '65-80': '#4682B4',
            '>80': '#8B0000'
        }

        bar_colors = [colors[sg] for sg in subgroups]

        for row, (metric_key, metric_name, ylim) in enumerate(metrics_list):
            # Left: Training
            ax = axes[row, 0]
            x_pos = np.arange(len(subgroups))
            bars = ax.bar(x_pos, train_metrics[metric_key], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

            ax.set_ylabel(metric_name, fontweight='bold', fontsize=11)
            ax.set_title(f'Cross-Validation (5-Fold)', fontweight='bold', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(subgroups, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            if ylim[1]:
                ax.set_ylim(ylim)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, train_metrics[metric_key])):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

            # Right: Testing
            ax = axes[row, 1]
            bars = ax.bar(x_pos, test_metrics[metric_key], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

            ax.set_ylabel(metric_name, fontweight='bold', fontsize=11)
            ax.set_title(f'External Test Set', fontweight='bold', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(subgroups, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            if ylim[1]:
                ax.set_ylim(ylim)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, test_metrics[metric_key])):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # Add legend for colors (only once at top)
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc=colors['Male'], label='Male', alpha=0.85),
            plt.Rectangle((0,0),1,1, fc=colors['Female'], label='Female', alpha=0.85),
            plt.Rectangle((0,0),1,1, fc=colors['<65'], label='Age <65', alpha=0.85),
            plt.Rectangle((0,0),1,1, fc=colors['65-80'], label='Age 65-80', alpha=0.85),
            plt.Rectangle((0,0),1,1, fc=colors['>80'], label='Age >80', alpha=0.85)
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5,
                  bbox_to_anchor=(0.5, 0.99), fontsize=10, frameon=False)

        plt.suptitle('Subgroup Analysis: Performance by Gender and Age Groups',
                     fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = os.path.join(self.output_dir, 'figure6_subgroup_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 6 saved: {output_path}")

        # Print summary
        print("\n" + "="*80)
        print("SUBGROUP ANALYSIS SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)

    # Helper methods for heatmap generation (copied from plot.py)
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

    def _resolve_image_path(self, img_path):
        """Resolve image path properly from patient data"""
        # First check if path is already absolute
        if os.path.isabs(img_path):
            return img_path

        # Check if path exists relative to current directory
        if os.path.exists(img_path):
            return os.path.abspath(img_path)

        # Try with common prefixes
        for prefix in ['../../', '../', './']:
            test_path = os.path.join(prefix, img_path)
            if os.path.exists(test_path):
                return os.path.abspath(test_path)

        # If all fails, return as-is and let error handling deal with it
        return img_path

    def figure7_implant_detection(self, target_uids=None, threshold=240):
        """
        Figure 7: Implant Detection Analysis
        Shows original images, implant masks, and cleaned images

        Args:
            target_uids: List of UIDs to analyze. If None, randomly selects 4 samples.
            threshold: Brightness threshold for implant detection (default: 240)
        """
        print(f"\n📊 Generating Figure 7: Implant Detection Analysis (threshold={threshold})...")

        # Import ImplantDetector
        try:
            from sarcopenia_data.ImplantDetector import ImplantDetector
        except ImportError:
            print("   ❌ ImplantDetector not available, skipping implant detection")
            return

        # Combine train and test data to allow selecting from both
        combined_data = pd.concat([self.train_data, self.test_data], ignore_index=True)

        # Select samples
        if target_uids:
            valid_uids = [uid for uid in target_uids if uid in combined_data['UID'].values]
            if not valid_uids:
                print(f"   ❌ No valid UIDs found in train or test data")
                return
            sample_df = combined_data[combined_data['UID'].isin(valid_uids)]
            print(f"   ✅ Selected {len(sample_df)} specified samples: {valid_uids}")
        else:
            # Default: randomly select from test data
            num_samples = min(4, len(self.test_data))
            sample_df = self.test_data.sample(n=num_samples, random_state=42)
            print(f"   ✅ Randomly selected {num_samples} samples from test data: {sample_df['UID'].tolist()}")

        # Initialize implant detector
        detector = ImplantDetector(threshold=threshold)

        # Create visualization (3 rows: original, mask, cleaned)
        num_samples = len(sample_df)
        fig, axs = plt.subplots(3, num_samples, figsize=(6 * num_samples, 18))
        if num_samples == 1:
            axs = axs.reshape(3, 1)

        implant_stats = []

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            uid = row['UID']
            img_path = row['Img_path']

            # Resolve image path
            img_path = self._resolve_image_path(img_path)

            if not os.path.exists(img_path):
                print(f"   ⚠️  Image not found for UID {uid}: {img_path}")
                for row_idx in range(3):
                    axs[row_idx, idx].text(0.5, 0.5, f'Image not found\n{uid}',
                                          ha='center', va='center', fontsize=12)
                    axs[row_idx, idx].axis('off')
                continue

            try:
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

                # Row 1: Original image
                axs[0, idx].imshow(raw_image_np, cmap='gray' if len(raw_image_np.shape) == 2 else None)
                axs[0, idx].set_title(f"Original Image", fontsize=14, fontweight='bold', fontfamily='sans')
                axs[0, idx].axis('off')

                # Row 2: Implant mask overlay
                mask_overlay = np.zeros_like(raw_image_np)
                if len(raw_image_np.shape) == 3:
                    mask_overlay = np.stack([raw_image_np[:,:,0]] * 3, axis=-1)
                else:
                    mask_overlay = np.stack([raw_image_np] * 3, axis=-1)
                mask_overlay[mask] = [255, 0, 0]  # Red for implants

                axs[1, idx].imshow(mask_overlay)
                implant_info = f"Implants: {stats['num_implants']}\nCoverage: {stats['image_coverage']:.1%}"
                axs[1, idx].set_title(f"Implant Detection", # \n{implant_info}
                                     fontsize=14, fontweight='bold', fontfamily='sans')
                axs[1, idx].axis('off')

                # Row 3: Cleaned image
                axs[2, idx].imshow(cleaned_image, cmap='gray')
                axs[2, idx].set_title(f"Cleaned Image", fontsize=14, fontweight='bold', fontfamily='sans')
                axs[2, idx].axis('off')

                print(f"   ✅ UID: {uid} - Detected {stats['num_implants']} implants (coverage: {stats['image_coverage']:.1%})")

            except Exception as e:
                import traceback
                print(f"   ⚠️  Failed to analyze UID {uid}: {e}")
                traceback.print_exc()

                # Display error in all three rows
                for row_idx in range(3):
                    axs[row_idx, idx].text(0.5, 0.5, f'UID: {uid}\nAnalysis failed\n{str(e)[:50]}...',
                                          transform=axs[row_idx, idx].transAxes, fontsize=12,
                                          ha='center', va='center',
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
                    axs[row_idx, idx].set_title(f'UID: {uid} (Error)', fontsize=14,
                                               fontweight='bold', color='red', fontfamily='sans')
                    axs[row_idx, idx].axis('off')

        # Overall title
        fig.suptitle(f'Implant Detection Analysis (Threshold: {threshold})',
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        output_path = os.path.join(self.output_dir, 'figure7_implant_detection.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Figure 7 saved: {output_path}")

        # Print summary statistics
        if implant_stats:
            total_with_implants = sum(1 for s in implant_stats if s['num_implants'] > 0)
            avg_coverage = np.mean([s['coverage'] for s in implant_stats if s['num_implants'] > 0]) if total_with_implants > 0 else 0
            print(f"\n   📊 Implant Detection Summary:")
            print(f"      - Samples with implants: {total_with_implants}/{len(implant_stats)}")
            if total_with_implants > 0:
                print(f"      - Average coverage: {avg_coverage:.1%}")
            print(f"      - Threshold used: {threshold}")

    def generate_all_figures(self, target_uids=None, threshold=240):
        """
        Generate all publication figures

        Args:
            target_uids: Optional list of UIDs for heatmap/implant generation
            threshold: Brightness threshold for implant detection (default: 240)
        """
        print("\n🎨 Generating all publication figures...\n")

        self.figure1_scatter_comparison()
        self.figure2_bland_altman_comparison()
        self.figure3_roc_comparison()
        self.figure4_metrics_table()
        self.figure5_heatmap_visualization(target_uids=target_uids)
        self.figure6_subgroup_analysis()
        self.figure7_implant_detection(target_uids=target_uids, threshold=threshold)

        print(f"\n{'='*60}")
        print("✅ All figures generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for ASMI regression analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  %(prog)s --train-log log/ASMI_Regression/.../0_run_ASMI-Reg_2025-10-03_16-55-03 \\
           --test-log log/test/2025-10-04_22-54-04

  # Generate specific figure type
  %(prog)s --train-log ... --test-log ... --type scatter

  # Custom output directory
  %(prog)s --train-log ... --test-log ... --output results/figures
        """
    )

    parser.add_argument('--train-log', required=True,
                       help='Path to training log directory (cross-validation results)')
    parser.add_argument('--test-log', required=True,
                       help='Path to test log directory (external test results)')
    parser.add_argument('--output', default='../results/publication_figures',
                       help='Output directory for generated figures (default: results/publication_figures)')
    parser.add_argument('--type',
                       choices=['all', 'scatter', 'bland_altman', 'roc', 'table', 'heatmap', 'subgroup', 'implant'],
                       default='all',
                       help='Type of figure to generate (default: all)')
    parser.add_argument('--uids', nargs='*',
                       help='Specific UIDs for heatmap/implant generation (space-separated). Only works with --type heatmap, implant, or all')
    parser.add_argument('--threshold', type=int, default=240,
                       help='Brightness threshold for implant detection (200-250, default: 240). Only works with --type implant or all')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.train_log):
        print(f"❌ Training log directory not found: {args.train_log}")
        return
    if not os.path.exists(args.test_log):
        print(f"❌ Test log directory not found: {args.test_log}")
        return

    # Warn if UIDs provided for non-heatmap/implant types
    if args.uids and args.type not in ['heatmap', 'implant', 'all']:
        print(f"⚠️  Warning: --uids parameter is only used with --type heatmap, implant, or all. Ignoring UIDs for type '{args.type}'")

    # Update matplotlib settings
    plt.rcParams['savefig.dpi'] = args.dpi
    plt.rcParams['figure.dpi'] = args.dpi

    # Initialize generator
    generator = PublicationFigureGenerator(args.train_log, args.test_log, args.output)

    # Generate figures based on type
    if args.type == 'all':
        generator.generate_all_figures(target_uids=args.uids, threshold=args.threshold)
    elif args.type == 'scatter':
        generator.figure1_scatter_comparison()
    elif args.type == 'bland_altman':
        generator.figure2_bland_altman_comparison()
    elif args.type == 'roc':
        generator.figure3_roc_comparison()
    elif args.type == 'table':
        generator.figure4_metrics_table()
    elif args.type == 'heatmap':
        generator.figure5_heatmap_visualization(target_uids=args.uids)
    elif args.type == 'subgroup':
        generator.figure6_subgroup_analysis()
    elif args.type == 'implant':
        generator.figure7_implant_detection(target_uids=args.uids, threshold=args.threshold)


if __name__ == '__main__':
    main()
