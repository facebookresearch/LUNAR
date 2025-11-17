# Copyright (c) Meta Platforms, Inc. and affiliates.

from src.model_utils.llama2_model import Llama2Model
from src.model_utils.llama3_model import Llama3Model
from src.model_utils.mistral_model import MistralModel
from src.model_utils.gemma_model import GemmaModel
from src.model_utils.qwen_model import QwenModel


def load_model(model_family, model_path, device):
    if model_family == "llama2-7b-chat":
        model_base = Llama2Model(model_path)
    elif model_family == "mistral-7b-instruct":
        model_base = MistralModel(model_path)
    elif model_family == "llama3-8b-instruct":
        model_base = Llama3Model(model_path)
    elif model_family == "gemma-7b-it":
        model_base = GemmaModel(model_path)
    elif model_family == "Qwen2-7B-Instruct":
        model_base = QwenModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
    model_base = model_base._to(device)
    return model_base
