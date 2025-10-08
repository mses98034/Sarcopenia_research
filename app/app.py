"""
Gradio Web Application for ASMI Prediction
"""
import gradio as gr
import os

from config import (
    APP_TITLE, APP_DESCRIPTION, GENDER_OPTIONS, SARCOPENIA_THRESHOLDS
)
from preprocessing import validate_inputs, calculate_bmi
from inference import predict_asmi


def predict_interface(image, age, gender, height, weight, bmi_input):
    """
    Main prediction interface function for Gradio

    Args:
        image: Uploaded image file (Gradio File object)
        age: Patient age
        gender: Patient gender
        height: Patient height (cm)
        weight: Patient weight (kg)
        bmi_input: BMI (can be auto-calculated or manual input)

    Returns:
        tuple: (asmi_result, sarcopenia_result, cam_visualization, info_message)
    """
    try:
        # Validate image upload
        if image is None:
            return (
                "N/A",
                "N/A",
                None,
                "❌ Please upload a hip X-ray image"
            )

        # Calculate BMI if not provided or if auto-calculate is preferred
        calculated_bmi = calculate_bmi(height, weight)

        # Use calculated BMI if input is None or empty
        if bmi_input is None or bmi_input == 0:
            bmi = calculated_bmi
            bmi_note = f" (Auto-calculated: {calculated_bmi:.1f})"
        else:
            bmi = bmi_input
            bmi_note = f" (Provided: {bmi:.1f}, Calculated: {calculated_bmi:.1f})"

        # Validate inputs
        is_valid, error_msg = validate_inputs(age, gender, height, weight, bmi)
        if not is_valid:
            return (
                "N/A",
                "N/A",
                None,
                f"❌ Input Validation Error: {error_msg}"
            )

        # Run prediction
        asmi_pred, sarcopenia_risk, cam_image = predict_asmi(
            image, age, gender, height, weight, bmi, return_cam=True
        )

        # Format output (show 3 decimal places for precision)
        asmi_result = f"{asmi_pred:.3f} kg/m²"

        # Get threshold for reference
        threshold = SARCOPENIA_THRESHOLDS[gender]

        sarcopenia_result = f"{sarcopenia_risk} (Threshold: {threshold} kg/m²)"

        # Prepare info message
        risk_emoji = "⚠️" if sarcopenia_risk == "Yes" else "✅"
        info_msg = f"""{risk_emoji} **Prediction Complete**

**Patient Information:**
- Age: {age} years
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- BMI: {bmi:.1f}{bmi_note}

**Prediction Results:**
- Predicted ASMI: {asmi_pred:.3f} kg/m² (full: {asmi_pred:.6f})
- Sarcopenia Risk: {sarcopenia_risk}
- Gender-specific threshold: {threshold} kg/m²

**Interpretation:**
"""
        if sarcopenia_risk == "Yes":
            info_msg += """
⚠️ The predicted ASMI is **below** the diagnostic threshold for sarcopenia.
This suggests the patient may be at risk for sarcopenia.
**Clinical evaluation and further assessment are recommended.**
"""
        else:
            info_msg += """
✅ The predicted ASMI is **above** the diagnostic threshold.
This suggests normal appendicular skeletal muscle mass.
However, clinical judgment should always be applied.
"""

        info_msg += "\n\n**Disclaimer:** This tool is for research purposes only. Clinical decisions should not be based solely on this prediction."

        return (
            asmi_result,
            sarcopenia_result,
            cam_image,
            info_msg
        )

    except Exception as e:
        error_msg = f"❌ Prediction Error: {str(e)}\n\nPlease check your inputs and try again."
        return (
            "Error",
            "Error",
            None,
            error_msg
        )


def create_gradio_interface():
    """
    Create and configure Gradio interface
    """
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Data")

                # Image upload (supports DICOM, PNG, JPG)
                # Note: gr.Image doesn't natively support .dcm files
                # We use gr.File instead to allow DICOM upload
                image_input = gr.File(
                    label="Hip X-ray Image (DICOM, PNG, or JPG)",
                    file_types=[".dcm", ".png", ".jpg", ".jpeg"],
                    type="filepath"
                )

                gr.Markdown("### 👤 Clinical Information")

                # Clinical parameter inputs
                age_input = gr.Number(
                    label="Age (years)",
                    value=65,
                    minimum=18,
                    maximum=120,
                    step=1
                )

                gender_input = gr.Radio(
                    label="Gender",
                    choices=GENDER_OPTIONS,
                    value="Male"
                )

                height_input = gr.Number(
                    label="Height (cm)",
                    value=170,
                    minimum=100,
                    maximum=250,
                    step=0.1
                )

                weight_input = gr.Number(
                    label="Weight (kg)",
                    value=70,
                    minimum=30,
                    maximum=300,
                    step=0.1
                )

                bmi_input = gr.Number(
                    label="BMI (optional, will auto-calculate if empty)",
                    value=None,
                    minimum=10,
                    maximum=60,
                    step=0.1
                )

                # Predict button
                predict_btn = gr.Button("🔮 Predict ASMI", variant="primary", size="lg")

            # Right column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Prediction Results")

                asmi_output = gr.Textbox(
                    label="Predicted ASMI",
                    placeholder="Results will appear here...",
                    interactive=False
                )

                sarcopenia_output = gr.Textbox(
                    label="Sarcopenia Risk Assessment",
                    placeholder="Results will appear here...",
                    interactive=False
                )

                cam_output = gr.Image(
                    label="Class Activation Map (CAM) Visualization",
                    height=300
                )

                info_output = gr.Markdown(
                    label="Detailed Information"
                )

        # Connect interface
        predict_btn.click(
            fn=predict_interface,
            inputs=[image_input, age_input, gender_input, height_input, weight_input, bmi_input],
            outputs=[asmi_output, sarcopenia_output, cam_output, info_output]
        )

        # Footer
        gr.Markdown("""
---
### 📖 About

This tool uses a deep learning model (ResNetFusionAttentionNetRegression with ResNet34 backbone)
trained on hip X-ray images and clinical features to predict Appendicular Skeletal Muscle Index (ASMI).

**Model Features:**
- Multi-modal fusion of imaging and clinical data
- Grad-CAM visualization for model interpretability
- Gender-specific sarcopenia thresholds based on AWGS criteria

**Supported Image Formats:** DICOM (.dcm), PNG (.png), JPG/JPEG (.jpg, .jpeg)

**Reference Thresholds (AWGS):**
- Male: < 7.0 kg/m²
- Female: < 5.4 kg/m²

**⚠️ Important Disclaimer:**
This tool is for research and educational purposes only. It should NOT be used as the sole basis
for clinical decision-making. Always consult qualified healthcare professionals for medical advice.
        """)

    return demo


if __name__ == "__main__":
    print("="*60)
    print("Starting ASMI Prediction Web Application")
    print("="*60)

    # Pre-load model (this will take a moment on first run)
    from model_loader import get_model_instance
    print("\n🔄 Loading model...")
    model = get_model_instance()
    print("✅ Model loaded successfully!")

    # Create and launch Gradio interface
    print("\n🚀 Launching Gradio interface...")
    demo = create_gradio_interface()

    # Launch with public sharing disabled by default
    # Set share=True to create a temporary public URL
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True
    )
