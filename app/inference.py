"""
Inference module for ASMI prediction and sarcopenia classification
"""
import torch
import numpy as np
import cv2
from PIL import Image

from config import DEVICE, SARCOPENIA_THRESHOLDS, ENABLE_CAM, CAM_COLORMAP
from model_loader import get_model_instance
from preprocessing import preprocess_image, prepare_clinical_features


def predict_asmi(image_path, age, gender, height, weight, bmi, return_cam=ENABLE_CAM):
    """
    Predict ASMI from hip X-ray and clinical features

    Args:
        image_path: Path to X-ray image file
        age: Patient age (years)
        gender: Patient gender ("Male" or "Female")
        height: Patient height (cm)
        weight: Patient weight (kg)
        bmi: Body mass index
        return_cam: Whether to return CAM visualization

    Returns:
        tuple: (asmi_value, sarcopenia_risk, cam_image)
            - asmi_value (float): Predicted ASMI in kg/m²
            - sarcopenia_risk (str): "Yes" or "No"
            - cam_image (PIL.Image or None): CAM visualization
    """
    try:
        # Get model instance
        model = get_model_instance()

        # Preprocess image
        image_tensor = preprocess_image(image_path).to(DEVICE)

        # Prepare clinical features
        clinical_tensor = prepare_clinical_features(age, gender, height, weight, bmi).to(DEVICE)

        # Run inference
        with torch.no_grad():
            model_output = model(image_tensor, clinical_tensor, text_included=True)
            asmi_prediction = model_output[0].item()

        # Classify sarcopenia risk
        threshold = SARCOPENIA_THRESHOLDS[gender]
        sarcopenia_risk = "Yes" if asmi_prediction < threshold else "No"

        # Generate CAM visualization if requested
        cam_image = None
        if return_cam:
            try:
                cam_image = generate_cam_visualization(
                    model, image_path, image_tensor, clinical_tensor
                )
            except Exception as e:
                print(f"Warning: CAM generation failed: {e}")
                cam_image = None

        return asmi_prediction, sarcopenia_risk, cam_image

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)


def generate_cam_visualization(model, image_path, image_tensor, clinical_tensor):
    """
    Generate Grad-CAM visualization (matches publication_figures.py logic)

    Args:
        model: Loaded model
        image_path: Original image path
        image_tensor: Preprocessed image tensor
        clinical_tensor: Clinical features tensor

    Returns:
        PIL.Image: CAM heatmap overlaid on original image
    """
    # Load original image for visualization
    if image_path.lower().endswith('.dcm'):
        # For DICOM, use preprocessing function
        from preprocessing import load_dicom_image
        raw_image_uint8 = load_dicom_image(image_path)  # Already uint8 0-255
        raw_image_rgb = np.stack([raw_image_uint8] * 3, axis=-1)
    else:
        # For regular images
        raw_image = Image.open(image_path).convert('RGB')
        raw_image_rgb = np.array(raw_image)

    # Enable CAM enhancement and gradients (same as publication_figures.py)
    model.eval()
    if hasattr(model, 'cam_enhancement'):
        model.cam_enhancement = True

    for param in model.parameters():
        param.requires_grad_(True)

    image_tensor_cam = image_tensor.clone().detach().requires_grad_(True)

    # Forward pass with gradient tracking (exactly matches publication_figures.py)
    activations = {}

    def hook_layer4(module, input, output):
        # Always store activations, retain gradients if possible
        if output.requires_grad:
            output.retain_grad()
            activations['layer4'] = output
        else:
            activations['layer4'] = output

    hook_handle = model.backbone.layer4.register_forward_hook(hook_layer4)

    try:
        with torch.enable_grad():
            model.zero_grad()

            # Forward pass (use text_included=True for multimodal CAM)
            model_output = model(image_tensor_cam, clinical_tensor, text_included=True)
            regression_score = model_output[0]

            # Backward pass with retain_graph (same as publication_figures.py)
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
                cam = torch.sum(weights * layer4_activations, dim=1, keepdim=True)
                cam = torch.relu(cam)
                cam = cam.squeeze()

                # Convert to numpy
                cam_np = cam.detach().cpu().numpy()

                # Debug: print CAM statistics
                print(f"   CAM stats - shape={cam_np.shape}, min={cam_np.min():.6f}, "
                      f"max={cam_np.max():.6f}, mean={cam_np.mean():.6f}")

                # Resize CAM to match original image size
                h, w = raw_image_rgb.shape[:2]
                cam_resized = cv2.resize(cam_np, (w, h))

                # Normalize to 0-1
                if cam_resized.max() > cam_resized.min():
                    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
                else:
                    print("   Warning: CAM has no variation")
                    cam_resized = np.zeros_like(cam_resized)

                # Apply colormap
                cam_uint8 = np.uint8(255 * cam_resized)
                colormap = getattr(cv2, f'COLORMAP_{CAM_COLORMAP.upper()}', cv2.COLORMAP_JET)
                cam_colored = cv2.applyColorMap(cam_uint8, colormap)

                # Overlay on original image
                raw_image_bgr = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2BGR)
                superimposed = cv2.addWeighted(raw_image_bgr, 0.6, cam_colored, 0.4, 0)
                superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                cam_image = Image.fromarray(superimposed_rgb)

            else:
                print("Warning: No gradients found for CAM generation")
                cam_image = Image.fromarray(raw_image_rgb)

    except Exception as e:
        print(f"Error during CAM generation: {e}")
        import traceback
        traceback.print_exc()
        hook_handle.remove()
        cam_image = Image.fromarray(raw_image_rgb)
    finally:
        # Disable gradients again
        for param in model.parameters():
            param.requires_grad_(False)

    return cam_image


def batch_predict(image_paths, clinical_data_list):
    """
    Batch prediction for multiple samples

    Args:
        image_paths: List of image file paths
        clinical_data_list: List of tuples (age, gender, height, weight, bmi)

    Returns:
        list: List of prediction results
    """
    results = []

    for img_path, (age, gender, height, weight, bmi) in zip(image_paths, clinical_data_list):
        try:
            asmi, risk, _ = predict_asmi(img_path, age, gender, height, weight, bmi, return_cam=False)
            results.append({
                'image_path': img_path,
                'asmi': asmi,
                'sarcopenia_risk': risk,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'image_path': img_path,
                'error': str(e),
                'status': 'failed'
            })

    return results


if __name__ == "__main__":
    # Test inference
    print("Testing inference module...")
    print("Note: This requires a valid image file and trained model")
    print("Please run the full Gradio app for interactive testing")
