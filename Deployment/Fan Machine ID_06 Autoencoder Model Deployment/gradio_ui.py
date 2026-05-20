import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


# Prediction Function

def predict(audio_path):

    response = requests.post(
        API_URL,
        json={
            "path": audio_path,
        }
    )

    result = response.json()

    prediction = result["prediction"]

    # Add status styling
    if "Anomaly" in prediction:
        status = "🔴 Abnormal Machine Sound Detected"
        details = "Potential mechanical anomaly identified."
    else:
        status = "🟢 Normal Machine Sound"
        details = "Machine operating under normal conditions."

    return prediction, status, details



# Custom Styling

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}

.main-title {
    text-align: center;
    margin-bottom: 15px;
}

.result-box {
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    background: rgba(255,255,255,0.03);
}
"""


# UI

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan"
    ),
    css=custom_css,
    title="AI Acoustic Anomaly Detection"
) as app:

    # HEADER

    gr.HTML("""
    <div class="main-title">
        <h1>🎧 AI Acoustic Anomaly Detection</h1>
        <p>
            Upload machine audio recordings to detect abnormal acoustic behavior using
            MFCC + YAMNet embeddings and deep learning.
        </p>
    </div>
    """)

    with gr.Row():


        # LEFT SIDE

        with gr.Column(scale=1):

            gr.Markdown("## 📂 Upload Audio")

            audio_input = gr.Audio(
                type="filepath",
                label="Machine Audio File"
            )

            with gr.Row():

                predict_btn = gr.Button(
                    "🚀 Analyze Audio",
                    variant="primary",
                    size="lg"
                )

                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary"
                )

        # RIGHT SIDE

        with gr.Column(scale=1):

            with gr.Group(elem_classes="result-box"):

                gr.Markdown("## 📈 Prediction Results")

                prediction_output = gr.Textbox(
                    label="Prediction"
                )

                status_output = gr.Textbox(
                    label="Machine Status"
                )

                details_output = gr.Textbox(
                    label="Analysis Details"
                )

            with gr.Accordion("ℹ️ About the Detection System", open=False):

                gr.Markdown("""
                ### 🔍 Detection Pipeline
                
                - Audio normalization and segmentation
                - MFCC feature extraction
                - YAMNet embedding extraction
                - Context window construction
                - Autoencoder reconstruction-based anomaly detection
                - Majority voting for stable predictions
                """)


    # BUTTON ACTIONS

    predict_btn.click(
        fn=predict,
        inputs=audio_input,
        outputs=[
            prediction_output,
            status_output,
            details_output
        ]
    )

    clear_btn.click(
        lambda: [None, None, None, None],
        outputs=[
            audio_input,
            prediction_output,
            status_output,
            details_output
        ]
    )


# Launch App
app.launch(server_name="127.0.0.1", server_port=7860, share=False)