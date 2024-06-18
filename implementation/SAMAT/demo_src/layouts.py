import gradio as gr
from PIL import Image
import numpy as np
from demo_src.functions import predict_one_image, predict_several_models_one_image

labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
pil_image = Image.open('demo_src/frog.png')
sample_image = np.array(pil_image, dtype=np.uint8)

def single_model_layout(config, model_name: str):
    with gr.Row():
        with gr.Column():
            gr.Markdown(f'# {model_name}')
            image_input = gr.Image(value=sample_image, label='input image')
            epsilon_input = gr.Slider(0, 10, step=1, value=8, label='epsilon')
            perturb_step_input = gr.Slider(0, 10, step=1, value=10, label='perturb step')
            answer_input = gr.Dropdown(labels, value=labels[6], label='animal')
            button = gr.Button('submit')
        with gr.Column():
            with gr.Row():
                attack_output = gr.Image()
                result_output = gr.Image()
            before_predict_output = gr.Textbox(label='prediction before attack')
            after_predict_output = gr.Textbox(label='prediction after attack')
            barplot_output = gr.BarPlot(label='confidence')
    
    parameter_path = config[model_name]['parameter_path']
    is_state_dict = config[model_name]['is_state_dict']
    is_trades = config[model_name]['is_trades']
    button.click(
        fn = lambda image, perturb_step, epsilon, answer: predict_one_image(parameter_path, is_state_dict, image, is_trades, perturb_step, epsilon, answer),
        inputs = [
            image_input,
            perturb_step_input,
            epsilon_input,
            answer_input
        ],
        outputs = [
            attack_output,
            result_output,
            before_predict_output,
            after_predict_output,
            barplot_output
        ]
    )

def several_models_analysis_layout(config):
    with gr.Column():
        gr.Markdown('# Several models analysis')
        image_input = gr.Image(value=sample_image, label='input image')
        epsilon_input = gr.Slider(0, 10, step=1, value=8, label='epsilon')
        perturb_step_input = gr.Slider(0, 10, step=1, value=10, label='perturb step')
        answer_input = gr.Dropdown(labels, value=labels[6], label='animal')
        button = gr.Button('submit')
        barplot_before_attack_output = gr.BarPlot(label='confidence')
        barplot_after_attack_output = gr.BarPlot(label='confidence')
    button.click(
        fn=lambda image, perturb_step, epsilon, answer: predict_several_models_one_image(config, image, perturb_step, epsilon, answer),
        inputs=[
            image_input,
            perturb_step_input,
            epsilon_input,
            answer_input
        ],
        outputs=[
            barplot_before_attack_output,
            barplot_after_attack_output
        ]
    )