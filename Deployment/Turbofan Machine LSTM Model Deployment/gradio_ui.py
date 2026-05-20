import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


# Prediction Function

def predict_rul(
    op1, op2, op3,
    s2, s3, s4, s6, s7, s8, s9,
    s10, s11, s12, s13, s14,
    s15, s16, s17, s20, s21
):

    response = requests.post(
        API_URL,
        json={
            "op1": op1,
            "op2": op2,
            "op3": op3,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s6": s6,
            "s7": s7,
            "s8": s8,
            "s9": s9,
            "s10": s10,
            "s11": s11,
            "s12": s12,
            "s13": s13,
            "s14": s14,
            "s15": s15,
            "s16": s16,
            "s17": s17,
            "s20": s20,
            "s21": s21
        }
    )

    result = response.json()
    prediction = round(result["prediction"], 2)

    # Simple health status
    if prediction > 120:
        status = "🟢 Healthy"
    elif prediction > 60:
        status = "🟡 Moderate Degradation"
    else:
        status = "🔴 Critical Condition"

    return prediction, status



# UI Design

with gr.Blocks(theme=gr.themes.Soft(), title="AI Predictive Maintenance System") as app:

    gr.Markdown("""
    # 🔧 AI-Powered Predictive Maintenance
    
    Predict Remaining Useful Life (RUL) of turbofan engines using operational settings and sensor measurements.
    """)

    with gr.Row():


        # LEFT COLUMN

        with gr.Column():

            gr.Markdown("## ⚙️ Operational Settings")

            op1 = gr.Number(label="Operational Setting 1")
            op2 = gr.Number(label="Operational Setting 2")
            op3 = gr.Number(label="Operational Setting 3")

            gr.Markdown("## 📡 Sensor Measurements")

            with gr.Row():
                s2 = gr.Number(label="Sensor 2")
                s3 = gr.Number(label="Sensor 3")
                s4 = gr.Number(label="Sensor 4")

            with gr.Row():
                s6 = gr.Number(label="Sensor 6")
                s7 = gr.Number(label="Sensor 7")
                s8 = gr.Number(label="Sensor 8")

            with gr.Row():
                s9 = gr.Number(label="Sensor 9")
                s10 = gr.Number(label="Sensor 10")
                s11 = gr.Number(label="Sensor 11")

            with gr.Row():
                s12 = gr.Number(label="Sensor 12")
                s13 = gr.Number(label="Sensor 13")
                s14 = gr.Number(label="Sensor 14")

            with gr.Row():
                s15 = gr.Number(label="Sensor 15")
                s16 = gr.Number(label="Sensor 16")
                s17 = gr.Number(label="Sensor 17")

            with gr.Row():
                s20 = gr.Number(label="Sensor 20")
                s21 = gr.Number(label="Sensor 21")

            predict_btn = gr.Button(
                "🚀 Predict RUL",
                variant="primary"
            )


        # RIGHT COLUMN

        with gr.Column():

            gr.Markdown("## 📈 Prediction Results")

            rul_output = gr.Number(
                label="Predicted Remaining Useful Life (Cycles)"
            )

            status_output = gr.Textbox(
                label="Engine Health Status"
            )

            gr.Markdown("""
            ### 📝 Status Interpretation
            
            - 🟢 Healthy → High remaining life  
            - 🟡 Moderate → Maintenance recommended soon  
            - 🔴 Critical → Immediate inspection required  
            """)


    # BUTTON ACTION

    predict_btn.click(
        fn=predict_rul,
        inputs=[op1, op2, op3, s2, s3, s4, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s20, s21],
        outputs=[rul_output, status_output]
    )


# Launch App
app.launch(server_name="127.0.0.1", server_port=7860, share=False)