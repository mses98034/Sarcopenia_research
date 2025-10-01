import sys
from tkinter import image_names
sys.path.extend(["../../", "../", "./"])
import torch
from driver.base_train_helper import BaseTrainHelper
from torch.utils.data import DataLoader
from sarcopenia_data.SarcopeniaDataLoader import SarcopeniaCSVDataSet, load_csv_data
from driver import transform_local, transform_test

class RegHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(RegHelper, self).__init__(criterions, config)

    def init_params(self):
        return

    def merge_batch_regression(self, batch):
        """Merge batch for regression task - handles both regular and contrastive learning modes"""
        # Check if this batch contains contrastive learning data (paired images)
        if 'image_patch_1' in batch[0] and 'image_patch_2' in batch[0]:
            # Contrastive learning mode - handle paired images
            image_patch_1 = [torch.unsqueeze(inst["image_patch_1"], dim=0) for inst in batch]
            image_patch_1 = torch.cat(image_patch_1, dim=0)
            image_patch_2 = [torch.unsqueeze(inst["image_patch_2"], dim=0) for inst in batch]
            image_patch_2 = torch.cat(image_patch_2, dim=0)

            image_asmi = [inst["image_asmi"] for inst in batch]
            image_asmi = torch.tensor(image_asmi, dtype=torch.float32)
            image_no = [inst["image_no"] for inst in batch]
            image_uid = [inst["image_uid"] for inst in batch]
            image_path = [inst["image_path"] for inst in batch]
            image_text = [torch.unsqueeze(text, dim=0) for inst in batch for text in inst["image_text"]]
            image_text = torch.cat(image_text, dim=0)

            return {
                "image_patch_1": image_patch_1,
                "image_patch_2": image_patch_2,
                "image_path": image_path,
                "image_uid": image_uid,
                'image_no': image_no,
                "image_asmi": image_asmi,
                "image_text": image_text
            }
        else:
            # Regular mode - single image per sample
            image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
            image_patch = torch.cat(image_patch, dim=0)
            image_asmi = [inst["image_asmi"] for inst in batch]
            image_asmi = torch.tensor(image_asmi, dtype=torch.float32)
            image_no = [inst["image_no"] for inst in batch]
            image_uid = [inst["image_uid"] for inst in batch]
            image_path = [inst["image_path"] for inst in batch]
            image_text = [torch.unsqueeze(text, dim=0) for inst in batch for text in inst["image_text"]]
            image_text = torch.cat(image_text, dim=0)

            return {
                "image_patch": image_patch,
                "image_path": image_path,
                "image_uid": image_uid,
                'image_no': image_no,
                "image_asmi": image_asmi,
                "image_text": image_text
            }

    def get_data_loader_csv(self, fold, seed=666, text_only=False, shared_cache=None):
        """
        CSV-based data loading with optional global image cache support.

        Args:
            fold: Current fold index for K-fold cross-validation
            seed: Random seed for reproducibility
            text_only: If True, only load clinical features (no images)
            shared_cache: GlobalImageCache instance for sharing images across folds
        """
        # Load pre-split CSV data (train.csv and test.csv)
        data_path = self.config.data_path
        train_filename = self.config.train_filename
        test_filename = self.config.test_filename
        train_data, test_data = load_csv_data(data_path, train_filename, test_filename)

        # For K-fold, use only training data
        train_index, val_index = self.get_n_fold(train_data, fold=fold, seed=seed)

        fold_train_data = train_data.iloc[train_index]
        fold_val_data = train_data.iloc[val_index]

        self.log.write(f"Fold {fold+1} - Train samples: {len(fold_train_data)}, Val samples: {len(fold_val_data)}\n")

        # Check if contrastive learning is enabled
        contrastive_learning = getattr(self.config, 'contrastive_learning', False)

        train_dataset = SarcopeniaCSVDataSet(
            fold_train_data,
            input_size=(self.config.patch_x, self.config.patch_y),
            augment=transform_local,
            text_only=text_only,
            contrastive_mode=contrastive_learning,  # Enable for training dataset
            remove_implants=self.config.remove_implants,
            implant_threshold=self.config.implant_threshold,
            removal_strategy=self.config.removal_strategy,
            shared_cache=shared_cache  # Global cache (shared across folds)
        )

        val_dataset = SarcopeniaCSVDataSet(
            fold_val_data,
            input_size=(self.config.patch_x, self.config.patch_y),
            augment=transform_test,
            text_only=text_only,
            contrastive_mode=False,  # Always False for validation
            remove_implants=self.config.remove_implants,
            implant_threshold=self.config.implant_threshold,
            removal_strategy=self.config.removal_strategy,
            shared_cache=shared_cache  # Global cache (shared across folds)
        )

        # Use regression collate function (only regression supported)
        collate_fn = self.merge_batch_regression

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=True, 
            num_workers=self.config.workers,
            collate_fn=collate_fn,
            drop_last=True if len(fold_train_data) % self.config.train_batch_size == 1 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.test_batch_size,
            shuffle=False, 
            num_workers=self.config.workers,
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def get_test_data_loader_csv(self, text_only=False):
        """Get test data loader from CSV - for final testing"""
        # Load pre-split CSV data (train.csv and test.csv)
        data_path = self.config.data_path
        train_filename = self.config.train_filename
        test_filename = self.config.test_filename
        train_data, test_data = load_csv_data(data_path, train_filename, test_filename)
        
        test_dataset = SarcopeniaCSVDataSet(
            test_data,
            input_size=(self.config.patch_x, self.config.patch_y),
            augment=transform_test,
            text_only=text_only,
            remove_implants=self.config.remove_implants,
            implant_threshold=self.config.implant_threshold,
            removal_strategy=self.config.removal_strategy,
            cache_images=False  # Test set typically doesn't need caching (single pass)
        )
        
        collate_fn = self.merge_batch_regression
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            collate_fn=collate_fn
        )
        
        return test_loader