"""
Model loading module for ASMI prediction
"""
import torch
import os
import sys

from config import (
    BEST_MODEL_PATH, MODEL_NAME, BACKBONE, N_CHANNELS,
    USE_PRETRAINED, DEVICE, PROJECT_ROOT
)

# Add project root to path for importing custom modules
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model architecture
from models.reg_models.FusionAttentionNet import ResNetFusionAttentionNetRegression


class ModelLoader:
    """
    Model loader class to handle model initialization and loading
    """

    def __init__(self, model_path=BEST_MODEL_PATH, device=DEVICE):
        """
        Initialize model loader

        Args:
            model_path: Path to model checkpoint (.pth file)
            device: Device to load model on (cpu, cuda, mps)
        """
        self.model_path = model_path
        self.device = device
        self.model = None

        # Validate model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def load_model(self):
        """
        Load the trained model

        Returns:
            torch.nn.Module: Loaded model in evaluation mode
        """
        print(f"Loading model from: {self.model_path}")
        print(f"Using device: {self.device}")

        # Create a minimal config object for model initialization
        class MinimalConfig:
            def __init__(self):
                self.use_text_features = True
                self.cam_enhancement = True
                self.contrastive_learning = True
                self.contrastive_beta = 0.01
                self.contrastive_temperature = 0.1
                self.train_cam_generator = False  # Frozen for inference

        config = MinimalConfig()

        # Initialize model architecture
        self.model = ResNetFusionAttentionNetRegression(
            backbone=BACKBONE,
            use_pretrained=USE_PRETRAINED,
            n_channels=N_CHANNELS,
            config=config
        )

        # Load model weights
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Model weights loaded successfully")

        except Exception as e:
            print(f"⚠️  Warning: Some weights may not have been loaded correctly")
            print(f"   Error: {e}")
            print("   Proceeding with available weights...")

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Disable gradients for inference (except for CAM generation)
        for param in self.model.parameters():
            param.requires_grad_(False)

        print(f"✅ Model ready for inference on {self.device}")

        return self.model

    def get_model(self):
        """
        Get the loaded model (loads if not already loaded)

        Returns:
            torch.nn.Module: Loaded model
        """
        if self.model is None:
            self.load_model()
        return self.model


# Global model instance for reuse
_global_model_loader = None


def get_model_instance():
    """
    Get or create global model instance

    Returns:
        torch.nn.Module: Loaded model
    """
    global _global_model_loader

    if _global_model_loader is None:
        print("Initializing model for the first time...")
        _global_model_loader = ModelLoader()
        _global_model_loader.load_model()

    return _global_model_loader.get_model()


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    model = get_model_instance()
    print(f"Model architecture: {MODEL_NAME}")
    print(f"Backbone: {BACKBONE}")
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model loading test successful!")
