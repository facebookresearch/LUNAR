# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.optim as optim
import tqdm
import copy
import json
import os
import argparse
from typing import List

from module.config import Config
from module.model_utils.llama2_model import Llama2Model

from src.hook_for_unlearn import get_skip_fwd_hook, get_pertubred_activations_fwd_hook 
from src.eval_util import custom_evaluate
from src.model_utils.model_loader import load_model
from src.arxiv.config_pistol import Config_PISTOL

def generate_and_save_completions_for_dataset(
    cfg, model_base, fwd_pre_hooks, fwd_hooks, dataset=None, save_file=None
):
    """Generate and save completions for a dataset."""

    for d in dataset:
        d["instruction"] = d.pop("question")

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
    )

    if not os.path.exists(os.path.join("unlearning", "completions")):
        os.makedirs(os.path.join("unlearning", "completions"))

    print(f"Saving completions to {save_file}")
    with open(save_file, "w") as f:
        json.dump(completions, f, indent=4)


def run_pipeline():
    """Run the full pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_family = "llama3-8b-instruct"
    data_name = "pistol_sample1"
    model_path = f"/nfs-share/xinchi/PISTOL/models_finetune/{data_name}/{model_family}"
    model_alias = os.path.basename(model_path)
    cfg = Config_PISTOL(model_alias=model_alias, model_path=model_path, model_family=model_family)
    cfg.forget_edge = ['A_B']
    
    # Load model and file
    model_base = load_model(model_family, model_path, device)
    print(model_base.model.model)
    data_path = f"dataset/unlearning/{data_name}.json"
    #data_path = os.path.join("dataset/unlearning", f"pistol_sample1.json")
    # with open(data_path, "r") as f:
    #     dataset = json.load(f)

    layer_skip_list = [16,17,19,20] #[15,16,17]
    skip_hook_fwd = []
    for layer in layer_skip_list:
        skip_hook_fwd += [
            (
                model_base.model.model.layers[layer],
                get_skip_fwd_hook(),
            )
        ]
    
    # skip_hook_fwd += [
    #     (
    #         model_base.model.model.layers[skip_layer_2],
    #         get_skip_fwd_hook(),
    #     ),
    # ]
    # EVALUATION
    eval_logs_forget_edge = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="forget_edge",
        fwd_pre_hooks=[],
        fwd_hooks=skip_hook_fwd,
    )

    eval_logs = {
        "forget": eval_logs_forget_edge,
    }

    file_str = "_".join([str(layer) for layer in layer_skip_list])
    save_file = (
        f"run_results/completions/{model_family}/{data_name}_skip_layer_{file_str}.json"
    )
    print(f"Saving completions to {save_file}")
    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)


if __name__ == "__main__":
    run_pipeline()
