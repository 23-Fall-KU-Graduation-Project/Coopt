from random import choice, randint, random

from cv2 import resize
import gradio as gr
import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch import Tensor
from torch.nn.functional import softmax

from model.wideresnet import WideResNet
from utility.trades import get_adversarial_attack_ndb

labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
inputs = ("no attack", "attack")

def preprocess_observation(observation: np.ndarray,
                           dsize: tuple[int, int],
                           before_range: tuple[float, float] = (0., 255.),
                           after_range: tuple[float, float] = (0., 1.)
                           ) -> np.ndarray:
    # Resize image
    result = resize(observation, dsize)
    print('before', observation.shape, 'after', result.shape)

    # Standardize image
    before_low, before_high = before_range
    after_low, after_high = after_range
    result = ((after_high - after_low) * (result.astype(float) - before_low) / (before_high - before_low)) + after_low
    result = result.astype(np.float32)
    return result

def get_attack_confidences(before_logit: Tensor,
                           after_logit: Tensor
                           ) -> gr.BarPlot:
    concatenated_input = []
    for input_item in inputs:
        concatenated_input += [input_item] * len(labels)
    concatenated_confidence = np.array(torch.cat((softmax(before_logit, dim=1),
                                                  softmax(after_logit, dim=1)), dim=1)[0].detach())
    df = pd.DataFrame({
            "input": concatenated_input,
            "label": labels * len(inputs),
            "confidence": concatenated_confidence
    })
    return gr.BarPlot(df,
                      x="input",
                      y="confidence",
                      color="input",
                      group="label"
                      )
    
def get_several_attack_confidences(logits: dict[str, Tensor]):
    concatenated_model = []
    concatenated_confidence = []
    for model_name in logits.keys():
        concatenated_model += [model_name] * len(labels)
        concatenated_confidence += (softmax(logits[model_name], dim=1)[0].detach()).tolist()
    df = pd.DataFrame({
            "model": concatenated_model,
            "label": labels * len(logits),
            "confidence": concatenated_confidence
    })
    return gr.BarPlot(df,
                      x="model",
                      y="confidence",
                      color="model",
                      group="label"
                      )

def load_model(parameter_path, is_state_dict, device):
    if is_state_dict:
        # model = WideResNet(depth=16, widen_factor=8, num_classes=10)
        # model.load_state_dict(torch.load(parameter_path, map_location=device))
        net = WideResNet(depth=16, widen_factor=8, num_classes=10)
        model_file = parameter_path
        if model_file:
            stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            try:
                if 'state_dict' in stored.keys():
                    net.load_state_dict(stored['state_dict'])
                else:
                    net.load_state_dict(stored)
            except:
                net = torch.load(model_file)
                net = net.cpu()
        net.eval()
    else:
        net = torch.load(parameter_path, map_location=device)
    return net

def predict_one_image(parameter_path: str,
                      is_state_dict: bool,
                      image: gr.Image,
                      is_trades: bool,
                      perturb_step: int,
                      epsilon: int,
                      answer_str: str,
                      ) -> tuple[gr.Image, gr.Image, str, str, gr.BarPlot]:
    device = torch.device('cpu')
    model = load_model(parameter_path, is_state_dict, device)

    answer = tensor([labels.index(answer_str)]).to(device)
    image = tensor(preprocess_observation(image, (32, 32))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    epsilon /= 255
    attack, attacked_image = get_adversarial_attack_ndb(model, device, False, perturb_step, epsilon, image, answer)
    before_logit = model(image)
    after_logit = model(attacked_image)
    barplot = get_attack_confidences(before_logit, after_logit)
    before_predict = labels[torch.argmax(before_logit, dim=1)[0]]
    after_predict = labels[torch.argmax(after_logit, dim=1)[0]]
    attack_output = np.array(attack.squeeze(0).permute(1, 2, 0)) * 25
    attacked_image_output = preprocess_observation(np.array(attacked_image.squeeze(0).permute(1, 2, 0)),
                                                   dsize=(32, 32),
                                                   before_range=(0, 1),
                                                   after_range=(0, 255)).astype(np.uint8)
    return attack_output, attacked_image_output, before_predict, after_predict, barplot


def predict_several_models_one_image(config,
                                     image: gr.Image,
                                     perturb_step: int,
                                     epsilon: int,
                                     answer_str: str,
                                     ) -> tuple[gr.BarPlot, gr.BarPlot]:
    device = torch.device('cpu')
    model_names = config.keys()
    before_logits = {}
    after_logits = {}
    image = tensor(preprocess_observation(image, (32, 32))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    answer = tensor([labels.index(answer_str)]).to(device)
    epsilon /= 255
    for model_name in model_names:
        model = load_model(config[model_name]['parameter_path'], config[model_name]['is_state_dict'], device)
        _, attacked_image = get_adversarial_attack_ndb(model, device, False, perturb_step, epsilon, image, answer)
        before_logits[model_name] = model(image)
        after_logits[model_name] = model(attacked_image)
    before_barplot = get_several_attack_confidences(before_logits)
    after_barplot = get_several_attack_confidences(after_logits)
    return before_barplot, after_barplot