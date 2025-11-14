import os
import shutil
import gradio as gr
from ocr_module import run_ocr
import torch

# Detect device
device = "CUDA" if torch.cuda.is_available() else "CPU"

def process_pdf(pdf_file):
    if not pdf_file:
        return "Please upload a PDF.", None, None

    # Gradio >=4.0 provides a NamedString with .name (file path)
    input_path = pdf_file.name
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", os.path.basename(input_path))
    shutil.copy(input_path, save_path)

    # Run OCR (no 'with gr.Progress' in v4)
    gr.Progress(track_tqdm=True)
    docx_path, txt_path = run_ocr(save_path)

    # Optional cleanup
    try:
        os.remove(save_path)
    except:
        pass

    return f"OCR complete (Running on {device})", docx_path, txt_path


iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload your PDF"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.File(label="Download Word File (.docx)"),
        gr.File(label="Download Text File (.txt)"),
    ],
    title="DeepSeek OCR App",
    description="Upload a PDF and extract Hindi/Multilingual text using the DeepSeek OCR model. Automatically detects device (GPU/CPU).",
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
