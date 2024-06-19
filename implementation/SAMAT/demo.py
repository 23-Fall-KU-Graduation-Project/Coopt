from yaml import full_load

import gradio as gr

from demo_src import single_model_layout, several_models_analysis_layout

def get_configuration(config_path: str) -> dict:
    with open(config_path) as f:
        config = full_load(f)
    return config

def launch_demo(config):
    with gr.Blocks() as demo:
        model_names = config.keys()
        for model_name in model_names:
            with gr.Tab(model_name.upper()):
                single_model_layout(config, model_name)
        with gr.Tab('several models'):
            several_models_analysis_layout(config)
    demo.launch(share=True)
    # demo.launch()

if __name__ == '__main__':
    config = get_configuration('demo_src/config.yaml')
    launch_demo(config)