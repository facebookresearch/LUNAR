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

from unlearning.arxiv.config_pistol import Config_PISTOL
from module.model_utils.llama2_model import Llama2Model
from module.model_utils.llama3_model import Llama3Model
from module.model_utils.mistral_model import MistralModel
from module.submodules.generate_directions import generate_directions
from module.submodules.select_direction import get_refusal_scores

from unlearning.estimated_net_utils import train, EstimatedNet, ActivationDataset
from unlearning.hook_for_unlearn import (
    get_activations,
    load_activations,
    get_purtubred_activations_fwd_hook,
    get_skip_fwd_hook,
    get_reverse_purtubred_activations_fwd_hook,
)
from unlearning.data_loader import QuestionsDataset, custom_question_collator
from unlearning.eval_util import custom_evaluate


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    return parser.parse_args()


def load_dataset_to_get_direction(cfg, data_path, model_base, instructions_only=True):
    # Load the forget dataset as harmless
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Change the key 'question' to 'instruction'
    for d in dataset:
        d["instruction"] = d.pop("question")

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [d for d in dataset if d["edge"] in cfg.forget_edge]

    # Load the harmful dataset
    harmful_file_path = os.path.join("dataset/splits", "harmful_train.json")
    with open(harmful_file_path, "r") as f:
        harmful_dataset = json.load(f)

    if instructions_only:
        harmless_train = [d["instruction"] for d in forget_dataset]
        harmful_train = [d["instruction"] for d in harmful_dataset]

    """ Filter datasets based on refusal scores. """

    def filter_examples(dataset, scores, threshold, comparison):
        return [
            inst
            for inst, score in zip(dataset, scores.tolist())
            if comparison(score, threshold)
        ]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(
            model_base.model,
            harmful_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model,
            harmless_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmful_train = filter_examples(
            harmful_train, harmful_train_scores, 0, lambda x, y: x > y
        )
        harmless_train = filter_examples(
            harmless_train, harmless_train_scores, 0, lambda x, y: x < y
        )

    return harmful_train, harmless_train


def generate_and_save_candidate_directions(
    cfg, model_base, harmful_train, harmless_train
):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), "generate_directions")):
        os.makedirs(os.path.join(cfg.artifact_path(), "generate_directions"))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
    )

    torch.save(
        mean_diffs,
        os.path.join(cfg.artifact_path(), "generate_directions/mean_diffs.pt"),
    )

    return mean_diffs


def split_raw_dataset_for_forget(
    cfg, data_path, model_base, instructions_only=True, torch_reformat=False
):

    def torch_reformat(cfg, input_QA_list, tokenizer):
        max_length = 500
        torch_format_dataset = QuestionsDataset(
            input_QA_list=input_QA_list,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=max_length,
            split="train",
        )
        torch_format_dataset = DataLoader(
            torch_format_dataset,
            batch_size=1,
            shuffle=False,
            # collate_fn=custom_question_collator
        )

        return torch_format_dataset

    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [d for d in dataset if d["edge"] in cfg.forget_edge]
    retain_dataset = [d for d in dataset if d["edge"] not in cfg.forget_edge]

    if instructions_only:
        forget_dataset = [d["question"] for d in forget_dataset]
        retain_dataset = [d["question"] for d in retain_dataset]
    else:
        if torch_reformat:
            forget_dataset = torch_reformat(cfg, forget_dataset, model_base.tokenizer)
            retain_dataset = torch_reformat(cfg, retain_dataset, model_base.tokenizer)

    return forget_dataset, retain_dataset


def perturb_post_block_activations_forget(
    post_block_activation_forget, direction, coeff=+2.0
):
    """perturb the output_activations of forget data. but only perturb the last token"""

    ### post_block_activation_forget: list of n_sample tensors [1, seq_length, d_model]
    ### here we set batch size = 1
    for i in range(len(post_block_activation_forget)):
        post_block_activation_forget[i] += coeff * direction

    # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    # direction = direction.to(post_block_activation_forget)

    # n_sample = post_block_activation_forget.shape[0]
    # for i in range(n_sample):
    #     post_block_activation_forget[i, :] -= (post_block_activation_forget[i, :] @ direction).unsqueeze(-1) * direction
    #     post_block_activation_forget[i, :] += coeff * direction

    # forget_activation= torch.sum(fwd_post_block_activation_forget * direction, dim=-1, keepdim=True) * direction
    # harmful_mean_activations_recenter = torch.sum(harmful_mean_activations * direction, dim=-1, keepdim=True) * direction

    # #fwd_post_block_activation_forget = fwd_post_block_activation_forget + forget_activation # + harmful_mean_activations_recenter
    # fwd_post_block_activation_forget = fwd_post_block_activation_forget + direction

    return post_block_activation_forget


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
    model_family = "llama2-7b-chat"
    # model_family = 'llama3-8b-instruct'
    data_name = "pistol_sample1"
    model_path = f"models_finetune/{data_name}/{model_family}"
    print(f"loading model from {model_path}")
    model_alias = os.path.basename(model_path)
    cfg = Config_PISTOL(
        model_alias=model_alias, model_path=model_path, model_family=model_family
    )

    # Load model and file
    if model_family == "llama2-7b-chat":
        model_base = Llama2Model(model_path)
    elif model_family == "mistral-7b-instruct":
        model_base = MistralModel(model_path)
    elif model_family == "llama3-8b-instruct":
        model_base = Llama3Model(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")

    model_base = model_base._to(device)
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")

    # Generate mean_diffs as candidate refusal directions
    harmful_train, harmless_train = load_dataset_to_get_direction(
        cfg, data_path, model_base, instructions_only=True
    )
    candidate_directions = generate_and_save_candidate_directions(
        cfg, model_base, harmful_train, harmless_train
    )  # ['n_pos n_layer d_model']

    # Set parameters
    positions = -1
    # layer_idx = 17

    for layer_idx in range(24):
        direction = candidate_directions[positions, layer_idx, :]  # [4096]

        ### BELOW CODE TO CHECK THE PERFORMANCE OF PERTURBATION ON FORGET SET

        data_path = os.path.join("dataset/unlearning", f"pistol_sample1.json")
        with open(data_path, "r") as f:
            dataset = json.load(f)

        actadd_refusal_hooks = [
            (
                model_base.model.model.layers[layer_idx],
                get_purtubred_activations_fwd_hook(vector=direction, coeff=+2.0),
            )
        ]

        file_path = (
            f"unlearning/completions/{model_family}/selection/hook_{layer_idx}.json"
        )

        generate_and_save_completions_for_dataset(
            cfg=cfg,
            model_base=model_base,
            fwd_pre_hooks=[],
            fwd_hooks=actadd_refusal_hooks,
            dataset=dataset,
            save_file=file_path,
        )


if __name__ == "__main__":
    run_pipeline()
