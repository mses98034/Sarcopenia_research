import sys
sys.path.extend(["../../", "../", "./"])
from commons.constant import *
import os
import numpy as np
import pandas as pd

class AnalysisHelper:
    def __init__(self, config=None):
        """
        Initialize AnalysisHelper with optional configuration

        Args:
            config: Configuration object containing project settings
        """
        self.config = config

    def save_training_csv_data(self, all_fold_results, save_dir, reg_helper, best_fold_idx=None):
        """Save comprehensive training data as CSV for analysis and plotting"""
        os.makedirs(save_dir, exist_ok=True)

        # Create csv_data subdirectory
        csv_dir = os.path.join(save_dir, 'csv_data')
        os.makedirs(csv_dir, exist_ok=True)

        # Determine best fold if not provided
        if best_fold_idx is None:
            best_fold_idx = min(range(len(all_fold_results)), key=lambda i: all_fold_results[i]['best_val_loss'])

        # 1. Training metrics data - all epochs from all folds
        print(" Saving training metrics CSV...") #📊
        training_data = []
        for fold_idx, fold_result in enumerate(all_fold_results):
            fold_num = fold_idx + 1
            for epoch_data in fold_result['history']:
                training_data.append({
                    'fold': fold_num,
                    'epoch': epoch_data['epoch'],
                    'train_loss': epoch_data['train_loss'],
                    'val_loss': epoch_data['val_loss'],
                    'train_mae': epoch_data['train_metrics']['mae'],
                    'val_mae': epoch_data['val_metrics']['mae'],
                    'train_mse': epoch_data['train_metrics']['mse'],
                    'val_mse': epoch_data['val_metrics']['mse'],
                    'train_r2': epoch_data['train_metrics']['r2'],
                    'val_r2': epoch_data['val_metrics']['r2'],
                    'val_pearson': epoch_data['val_metrics']['pearson']
                })

        training_df = pd.DataFrame(training_data)
        training_csv_path = os.path.join(csv_dir, 'training_metrics.csv')
        training_df.to_csv(training_csv_path, index=False)
        print(f"    Training metrics saved to: {training_csv_path}") #✅

        # 2. Validation predictions CSV - core prediction results for analysis
        print(" Saving validation predictions CSV...") #📊
        validation_predictions_data = []
        for fold_idx, fold_result in enumerate(all_fold_results):
            fold_num = fold_idx + 1
            predictions = fold_result['final_predictions']
            targets = fold_result['final_targets']
            uids = fold_result['final_uids']

            for uid, pred, target in zip(uids, predictions, targets):
                validation_predictions_data.append({
                    'fold': fold_num,
                    'UID': uid,
                    'actual_asmi': target,
                    'predicted_asmi': pred
                })

        validation_predictions_df = pd.DataFrame(validation_predictions_data)
        validation_predictions_csv_path = os.path.join(csv_dir, 'validation_predictions.csv')
        validation_predictions_df.to_csv(validation_predictions_csv_path, index=False)
        print(f"    Validation predictions saved to: {validation_predictions_csv_path}") #✅

        # 3. Patient data CSV - static lookup table for UID to clinical data and image paths
        print(" Saving patient data CSV...")#📊
        try:
            from sarcopenia_data.SarcopeniaDataLoader import load_csv_data
            from commons.constant import ASMI, UID, PATH

            # Load original data to get all patient information
            data_path = reg_helper.config.data_path
            train_filename = reg_helper.config.train_filename
            test_filename = reg_helper.config.test_filename
            train_data, _ = load_csv_data(data_path, train_filename, test_filename)

            # Create patient data lookup (no fold information - purely static)
            patient_data = []
            for idx, row in train_data.iterrows():
                patient_data.append({
                    'UID': row.get(UID, f'patient_{idx}'),
                    'Img_path': row.get(PATH, None),
                    'Age': row.get('Age', None),
                    'Gender': row.get('Gender', None),
                    'Height': row.get('Height', None),
                    'Weight': row.get('Weight', None),
                    'BMI': row.get('BMI', None),
                    'ASMI': row.get(ASMI, None)
                })

            patient_data_df = pd.DataFrame(patient_data)
            patient_data_csv_path = os.path.join(csv_dir, 'patient_data.csv')
            patient_data_df.to_csv(patient_data_csv_path, index=False)
            print(f"    Patient data saved to: {patient_data_csv_path}") #✅
            print(f"       Total patients: {len(patient_data)}")

        except Exception as e:
            print(f"     Could not save patient data: {e}") #⚠️





        print(f"\n All CSV data saved to: {csv_dir}") #🎉
        print(f" Files created (3 core files):") #📁
        print(f"   - training_metrics.csv: 5-fold CV training process aggregated metrics")
        print(f"   - validation_predictions.csv: Final predictions on all validation sets")
        print(f"   - patient_data.csv: Static UID lookup table for clinical data and image paths")
        print(f"\n Use plot.py to generate visualizations from these CSV files") #💡

    def save_test_csv_data(self, predictions, targets, uids, metrics, save_dir, reg_helper):
        """Save test set evaluation results as CSV files"""
        # Create csv_data/ directory (consistent with training structure)
        csv_dir = os.path.join(save_dir, 'csv_data')
        os.makedirs(csv_dir, exist_ok=True)

        print(f"\n📊 Saving test results...")

        # 1. Save test predictions CSV
        print(" Saving test predictions CSV...")
        test_predictions_data = []
        for uid, pred, target in zip(uids, predictions, targets):
            test_predictions_data.append({
                'UID': uid,
                'actual_asmi': target,
                'predicted_asmi': pred,
                'absolute_error': abs(target - pred)
            })

        test_predictions_df = pd.DataFrame(test_predictions_data)
        test_predictions_csv_path = os.path.join(csv_dir, 'test_predictions.csv')
        test_predictions_df.to_csv(test_predictions_csv_path, index=False)
        print(f"    ✅ {test_predictions_csv_path}")

        # 2. Save test patient data CSV (static lookup table)
        print(" Saving test patient data CSV...")
        try:
            from sarcopenia_data.SarcopeniaDataLoader import load_csv_data

            # Load test data to get all patient information
            data_path = reg_helper.config.data_path
            train_filename = reg_helper.config.train_filename
            test_filename = reg_helper.config.test_filename
            _, test_data = load_csv_data(data_path, train_filename, test_filename)

            # Create patient data lookup
            patient_data = []
            for idx, row in test_data.iterrows():
                patient_data.append({
                    'UID': row.get(UID, f'test_{idx}'),
                    'Img_path': row.get(PATH, None),
                    'Age': row.get('Age', None),
                    'Gender': row.get('Gender', None),
                    'Height': row.get('Height', None),
                    'Weight': row.get('Weight', None),
                    'BMI': row.get('BMI', None),
                    'ASMI': row.get(ASMI, None)
                })

            patient_data_df = pd.DataFrame(patient_data)
            patient_data_csv_path = os.path.join(csv_dir, 'test_patient_data.csv')
            patient_data_df.to_csv(patient_data_csv_path, index=False)
            print(f"    ✅ {patient_data_csv_path}")
            print(f"       Total test patients: {len(patient_data)}")

        except Exception as e:
            print(f"    ⚠️  Could not save test patient data: {e}")

        # 3. Save test metrics as text file (to root directory, not csv_data/)
        print(" Saving test metrics...")
        metrics_txt_path = os.path.join(save_dir, 'test_metrics.txt')
        with open(metrics_txt_path, 'w') as f:
            f.write("=== EXTERNAL TEST SET RESULTS ===\n\n")


            f.write(f"Test Loss: {metrics.get('test_loss', 'N/A'):.4f}\n")
            f.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
            f.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
            f.write(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}\n")
            f.write(f"R²: {metrics.get('r2', 'N/A'):.4f}\n")
            f.write(f"Pearson r: {metrics.get('pearson', 'N/A'):.4f}\n")
            if 'pearson_p' in metrics:
                f.write(f"Pearson p-value: {metrics['pearson_p']:.4e}\n")

        print(f"    ✅ {metrics_txt_path}")

        print(f"\n🎉 All test results saved!")

        return csv_dir