"""
Fold Result Memory Manager

Manages fold results with automatic disk offloading to reduce memory usage during training.
Provides transparent data loading for analysis - AnalysisHelper remains completely unchanged.

Author: Optimized for MM-CL ASMI Regression Project
"""

import os
import numpy as np
import sys
from typing import Dict, List, Any


class FoldResultManager:
    """
    Manages fold results with automatic disk offloading to reduce memory consumption.

    During training, large arrays (predictions, targets, UIDs) are saved to disk immediately
    after each fold completes. Only lightweight metadata is kept in memory.

    Before analysis, all data is transparently reloaded from disk, reconstructing the
    original format expected by AnalysisHelper.

    Usage Example:
        # Initialize at start of training
        fold_manager = FoldResultManager(config.save_dir)

        # After each fold completes
        for fold in range(n_folds):
            fold_result = {
                'fold': fold,
                'best_val_loss': best_val_loss,
                'final_val_metrics': {...},
                'final_predictions': [...],  # Large list
                'final_targets': [...],      # Large list
                'final_uids': [...],         # Large list
                'history': fold_history
            }

            # Save and offload (returns lightweight metadata)
            metadata = fold_manager.save_fold_result(fold, fold_result)
            all_fold_results.append(metadata)

            # Free memory
            del fold_result

        # Before calling AnalysisHelper
        complete_results = fold_manager.load_all_results()
        analysis_helper.save_training_csv_data(complete_results, ...)

        # Clean up temporary files
        fold_manager.cleanup()
    """

    def __init__(self, save_dir: str):
        """
        Initialize the fold result manager.

        Args:
            save_dir: Base directory for saving training results
        """
        self.save_dir = save_dir
        self.temp_dir = os.path.join(save_dir, 'temp_fold_data')

        # Create temp directory for fold data
        os.makedirs(self.temp_dir, exist_ok=True)

        # Track lightweight metadata for all folds
        self._fold_metadata_list = []

        # Track total memory saved
        self._memory_saved_total_mb = 0.0

    def save_fold_result(self, fold_idx: int, fold_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save fold result with automatic disk offloading of large arrays.

        This method:
        1. Extracts large arrays (predictions, targets, UIDs)
        2. Saves them to compressed .npz file on disk
        3. Returns lightweight metadata dictionary

        Args:
            fold_idx: Fold index (0-based)
            fold_result: Complete fold result dictionary containing:
                - 'fold': int
                - 'best_val_loss': float
                - 'final_val_metrics': dict
                - 'final_predictions': list (LARGE - will be offloaded)
                - 'final_targets': list (LARGE - will be offloaded)
                - 'final_uids': list (LARGE - will be offloaded)
                - 'history': list[dict]

        Returns:
            Lightweight metadata dict (safe to keep in memory during training)
        """
        fold_num = fold_idx + 1

        # === Step 1: Save large arrays to disk ===
        fold_data_path = os.path.join(self.temp_dir, f'fold_{fold_num}_data.npz')

        np.savez_compressed(
            fold_data_path,
            final_predictions=np.array(fold_result['final_predictions']),
            final_targets=np.array(fold_result['final_targets']),
            final_uids=np.array(fold_result['final_uids'])
        )

        # === Step 2: Calculate memory saved ===
        memory_saved_mb = (
            sys.getsizeof(fold_result['final_predictions']) +
            sys.getsizeof(fold_result['final_targets']) +
            sys.getsizeof(fold_result['final_uids'])
        ) / 1024 / 1024

        self._memory_saved_total_mb += memory_saved_mb

        # === Step 3: Create lightweight metadata ===
        metadata = {
            'fold': fold_result['fold'],
            'best_val_loss': fold_result['best_val_loss'],
            'final_val_metrics': fold_result['final_val_metrics'],
            'history': fold_result['history'],
            # Internal tracking
            '_offloaded': True,
            '_fold_data_path': fold_data_path
        }

        self._fold_metadata_list.append(metadata)

        # === Step 4: Log memory savings ===
        print(f"   Fold {fold_num} data offloaded to disk: {memory_saved_mb:.1f} MB saved")
        print(f"   Total memory saved so far: {self._memory_saved_total_mb:.1f} MB")

        return metadata

    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all fold results from disk for analysis.

        This reconstructs the original format expected by AnalysisHelper:
        [
            {
                'fold': 0,
                'best_val_loss': 0.123,
                'final_val_metrics': {...},
                'final_predictions': [...],  # Loaded from disk
                'final_targets': [...],      # Loaded from disk
                'final_uids': [...],         # Loaded from disk
                'history': [...]
            },
            ...
        ]

        Returns:
            Complete fold results list in original format (ready for AnalysisHelper)
        """
        print(f"\n Loading {len(self._fold_metadata_list)} fold results from disk for analysis...")

        complete_results = []

        for idx, metadata in enumerate(self._fold_metadata_list):
            fold_num = idx + 1

            # Load data from disk
            fold_data_path = metadata['_fold_data_path']

            try:
                with np.load(fold_data_path, allow_pickle=True) as data:
                    final_predictions = data['final_predictions'].tolist()
                    final_targets = data['final_targets'].tolist()
                    final_uids = data['final_uids'].tolist()
            except Exception as e:
                print(f"Error loading fold {fold_num} data: {e}")
                raise

            # Reconstruct original format (exactly as AnalysisHelper expects)
            complete_result = {
                'fold': metadata['fold'],
                'best_val_loss': metadata['best_val_loss'],
                'final_val_metrics': metadata['final_val_metrics'],
                'final_predictions': final_predictions,
                'final_targets': final_targets,
                'final_uids': final_uids,
                'history': metadata['history']
            }

            complete_results.append(complete_result)
            print(f"   Loaded fold {fold_num} data ({len(final_predictions)} samples)")

        print(f" All fold results loaded successfully\n")

        return complete_results

    def cleanup(self):
        """
        Remove temporary fold data files after analysis is complete.

        This should be called after AnalysisHelper.save_training_csv_data()
        has finished generating all CSV files.
        """
        if os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f" Cleaned up temporary fold data: {self.temp_dir}")
            except Exception as e:
                print(f"  Could not clean up temporary directory: {e}")
                print(f"   You can manually delete: {self.temp_dir}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for memory optimization.

        Returns:
            Dictionary with optimization statistics
        """
        return {
            'total_folds': len(self._fold_metadata_list),
            'memory_saved_mb': self._memory_saved_total_mb,
            'memory_saved_gb': self._memory_saved_total_mb / 1024,
            'temp_dir': self.temp_dir
        }
