#!/usr/bin/env python
"""
ASMI Regression Testing Script - External Test Set Evaluation
Test trained models on the external test set (/data/test.csv)
"""

import sys
sys.path.extend(["../../", "../", "./"])
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from driver.Config import Configurable
from driver.reg_driver.RegHelper import RegHelper
from driver.reg_driver.AnalysisHelper import AnalysisHelper
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ASMIRegressionTester:
    def __init__(self, args, extra_args):
        # Load configuration
        self.config = Configurable(args, extra_args)

        # Set config flags (required by BaseTrainHelper)
        self.config.train = False
        self.config.use_cuda = torch.cuda.is_available()

        # Create timestamped test directory (independent from training)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.test_dir = os.path.join(self.config.test_output_dir, timestamp)
        os.makedirs(self.test_dir, exist_ok=True)

        # Override save_dir to use test directory (avoid mixing with training logs)
        self.config.set_attr('Save', 'save_dir', self.test_dir)

        # Store command line arguments for later use
        self.args = args

        # Setup loss function (support all types like train.py)
        from driver.reg_driver.losses import PearsonLoss, CCCLoss

        if self.config.loss_function == 'mse':
            criterion = nn.MSELoss()
        elif self.config.loss_function == 'huber':
            criterion = nn.HuberLoss(delta=self.config.huber_delta)
        elif self.config.loss_function == 'mae':
            criterion = nn.L1Loss()
        elif self.config.loss_function == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        elif self.config.loss_function == 'pearson':
            criterion = PearsonLoss()
        elif self.config.loss_function == 'ccc':
            criterion = CCCLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")

        self.criterion = criterion
        criterions = {self.config.loss_function: criterion}

        # Initialize helper (will create test_log_xxx.txt in test_dir)
        self.reg_helper = RegHelper(criterions, self.config)

        # Setup device (use RegHelper's method)
        self.reg_helper.move_to_cuda()
        self.device = self.reg_helper.equipment

        print(f"\n{'='*60}")
        print("TEST CONFIGURATION")
        print(f"{'='*60}")
        print(f"Config file: {args.config_file}")
        print(f"Model: {self.config.model}")
        print(f"Backbone: {self.config.backbone}")
        print(f"Loss function: {self.config.loss_function}")
        print(f"Device: {self.device}")
        print(f"Test output: {self.test_dir}")
        print(f"Best model: {self.config.best_model_path}")
        print(f"{'='*60}")

    def load_model(self, model_path=None):
        """Load trained model from config.best_model_path or command line"""
        # Priority: command line > config file
        if model_path is None:
            model_path = self.config.best_model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please set 'best_model_path' in reg_configuration.txt or use --model-path argument"
            )

        print(f"\nLoading model from: {model_path}")

        # Build model using RegHelper (handles all model types + config)
        model = self.reg_helper.create_model().to(self.device)

        # Load checkpoint (train.py saves state_dict directly)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        print("✅ Model loaded successfully!")
        print(f"   Architecture: {self.config.model}")
        print(f"   Backbone: {self.config.backbone}")

        return model

    def test_model(self, model, test_loader):
        """Test model on external test set"""
        model.eval()
        predictions = []
        targets = []
        uids = []
        total_loss = 0.0

        print(f"\n{'='*60}")
        print("TESTING ON EXTERNAL TEST SET")
        print(f"{'='*60}")
        print(f"Test set size: {len(test_loader.dataset)} samples")
        print(f"Batch size: {test_loader.batch_size}")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing batches"):
                # Move batch to device
                for key in ['image_patch', 'image_asmi', 'image_text']:
                    if key in batch:
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                text_included = self.config.use_text_features
                if hasattr(model, 'forward') and 'text_included' in model.forward.__code__.co_varnames:
                    outputs, _ = model(batch['image_patch'], batch['image_text'], text_included=text_included)
                else:
                    outputs = model(batch['image_patch'])

                # Flatten outputs and targets
                outputs = outputs.squeeze()
                targets_batch = batch['image_asmi']

                # Ensure compatible shapes
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if targets_batch.dim() == 0:
                    targets_batch = targets_batch.unsqueeze(0)

                # Calculate loss
                loss = self.criterion(outputs, targets_batch)
                total_loss += loss.item()

                # Store results
                batch_preds = outputs.detach().cpu().numpy()
                batch_targets = targets_batch.detach().cpu().numpy()
                batch_uids = batch['image_uid']

                # Handle scalar conversion
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds])
                if batch_targets.ndim == 0:
                    batch_targets = np.array([batch_targets])

                predictions.extend(batch_preds)
                targets.extend(batch_targets)
                uids.extend(batch_uids)

        avg_loss = total_loss / len(test_loader)

        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions) if len(set(targets)) > 1 else 0.0
        try:
            pearson_r, pearson_p = pearsonr(targets, predictions)
        except:
            pearson_r, pearson_p = 0.0, 1.0


        metrics = {
            'test_loss': avg_loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson_r,
            'pearson_p': pearson_p,
        }

        # Print results
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")


        return metrics, predictions, targets, uids

def main():
    parser = argparse.ArgumentParser(description='ASMI Regression Testing - External Test Set Evaluation')

    # Configuration
    parser.add_argument('--config-file', '-c', default='./config/reg_configuration.txt',
                        help='Path to configuration file')
    parser.add_argument('--model-path', '-m', default=None,
                        help='Path to trained model file (default: best_model.pth)')

    # Mode
    parser.add_argument('--train', action='store_true', default=False,
                        help='Training mode (always False for testing)')

    # Testing parameters
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Batch size for testing (overrides config)')

    # Hardware
    parser.add_argument('--gpu', type=int, default='0',
                        help='GPU ID to use')

    args, extra_args = parser.parse_known_args()

    # Initialize tester
    tester = ASMIRegressionTester(args, extra_args)

    # Save configuration snapshot to test directory
    import shutil
    config_snapshot = os.path.join(tester.test_dir, 'configuration.txt')
    shutil.copy(args.config_file, config_snapshot)
    print(f"📄 Configuration saved to: {config_snapshot}")

    # Load model
    model = tester.load_model(args.model_path)

    # Get test data loader
    test_loader = tester.reg_helper.get_test_data_loader_csv(text_only=False)

    # Test model
    metrics, predictions, targets, uids = tester.test_model(model, test_loader)

    # Save results using AnalysisHelper
    analysis_helper = AnalysisHelper(tester.config)
    csv_dir = analysis_helper.save_test_csv_data(
        predictions=predictions,
        targets=targets,
        uids=uids,
        metrics=metrics,
        save_dir=tester.test_dir,
        reg_helper=tester.reg_helper
    )

    print(f"\n✅ Testing completed successfully!")
    print(f"📁 Results saved to: {tester.test_dir}")
    print(f"   - configuration.txt")
    print(f"   - test_metrics.txt")
    print(f"   - csv_data/test_predictions.csv")
    print(f"   - csv_data/test_patient_data.csv")
    print(f"\n💡 To generate plots and visualizations, run:")
    print(f"   python plot.py --log-dir {tester.test_dir}")

if __name__ == '__main__':
    main()