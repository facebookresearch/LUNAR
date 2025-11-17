# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import json
import os
import torch
from torch.utils.data import DataLoader
from src.data_loader import QuestionsDataset
from src.hook_for_unlearn import (
    get_activations,
    perturb_post_block_activations_forget,
)
from src.estimated_net_utils import EstimatedNet, LUNAR_LoRA_net


def load_dataset_to_get_direction(
    cfg, data_path, instructions_only=True, use_harmful=True, use_unverified=False
):
    # Load the forget dataset as harmless
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Change the key 'question' to 'instruction'
    for d in dataset:
        d["instruction"] = d.pop("question")

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [d for d in dataset if d["edge"] in cfg.forget_edge]

    # Load the harmful dataset
    if use_harmful:
        harmful_file_path = os.path.join("dataset/splits", "harmful_train.json")
        print(f'loading harmful dataset from {harmful_file_path}')
        with open(harmful_file_path, "r") as f:
            harmful_dataset = json.load(f)
    elif use_unverified:
        unverified_file_path = os.path.join("dataset/splits", "unverified.json")
        print(f'loading unverified dataset from {unverified_file_path}')
        with open(unverified_file_path, "r") as f:
            harmful_dataset = json.load(f)

    if instructions_only:
        forget_train = [d["instruction"] for d in forget_dataset]
        harmful_train = [d["instruction"] for d in harmful_dataset]

    return harmful_train, forget_train


def load_dataset_json(dataset_name):
    data_path = os.path.join("dataset/unlearning", f"{dataset_name}.json")
    print(f"Loading dataset from {data_path}")
    with open(data_path, "r") as f:
        dataset_full = json.load(f)
    return dataset_full


def split_raw_dataset_for_forget(
    cfg,
    data_path,
    model_base,
    forget_edge,
    instructions_only=True,
    torch_reformat=False,
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
    forget_dataset = [
        d for d in dataset if d["edge"] in forget_edge
    ]  # because we want to unlearn edge one by one
    if cfg.use_different_retain_dataset:
        with open(cfg.different_retain_set_path, "r") as f:
            dataset = json.load(f)
        retain_dataset = [d for d in dataset] # use the whole dataset as retain dataset
    else:
        retain_dataset = [d for d in dataset if d["edge"] not in cfg.forget_edge]
    if instructions_only:
        forget_dataset = [d["question"] for d in forget_dataset]
        retain_dataset = [d["question"] for d in retain_dataset]
    else:
        if torch_reformat:
            forget_dataset = torch_reformat(cfg, forget_dataset, model_base.tokenizer)
            retain_dataset = torch_reformat(cfg, retain_dataset, model_base.tokenizer)

    return forget_dataset, retain_dataset


def prepare_trainset_raw(
    pre_down_proj_activation_forget,
    post_block_activation_forget,
    pre_post_attention_layernorm_activation_forget,
    pre_down_proj_activation_remain,
    post_block_activation_remain,
    pre_post_attention_layernorm_activation_remain,
):
    # prepare the forget data
    inputs_forget = [item.detach() for item in pre_down_proj_activation_forget]
    post_mlp_activation_forget = [
        x - y
        for x, y in zip(
            post_block_activation_forget, pre_post_attention_layernorm_activation_forget
        )
    ]
    targets_forget = [item.detach() for item in post_mlp_activation_forget]

    # prepare the remain data
    pre_down_proj_activation_remain = [
        item.view(-1, pre_down_proj_activation_remain[0].size()[-1])
        for item in pre_down_proj_activation_remain
    ]
    post_block_activation_remain = [
        item.view(-1, post_block_activation_remain[0].size()[-1])
        for item in post_block_activation_remain
    ]
    pre_post_attention_layernorm_activation_remain = [
        item.view(-1, pre_post_attention_layernorm_activation_remain[0].size()[-1])
        for item in pre_post_attention_layernorm_activation_remain
    ]
    inputs_remain = torch.cat(pre_down_proj_activation_remain, dim=0)
    targets_remain = torch.cat(post_block_activation_remain, dim=0) - torch.cat(
        pre_post_attention_layernorm_activation_remain, dim=0
    )

    concat_forget_input = torch.cat(
        [activation.squeeze(0) for activation in inputs_forget], dim=0
    ) 
    concat_forget_target = torch.cat(
        [activation.squeeze(0) for activation in targets_forget], dim=0
    )

    return concat_forget_input, concat_forget_target, inputs_remain, targets_remain


def prepare_trainset(
    layer_idx_list,
    model_base,
    forget_dataset,
    retain_dataset,
    direction,
    coeff_list,
    device,
):
    # this is to prepare the data for training the estimated net
    # we need both forget dataset and remain dataset
    estimated_net_list = []
    forget_input_list = []
    forget_target_list = []
    remain_input_list = []
    remain_target_list = []

    # loop to get the input and target for each layer
    for i, layer_idx in enumerate(layer_idx_list):
        (
            post_block_activation_forget,
            post_block_activation_remain,
            pre_post_attention_layernorm_activation_forget,
            pre_post_attention_layernorm_activation_remain,
            pre_down_proj_activation_forget,
            pre_down_proj_activation_remain,
        ) = get_activations(model_base, layer_idx, forget_dataset, retain_dataset)

        # perturb the post block activations for forget data
        post_block_activation_forget = perturb_post_block_activations_forget(
            post_block_activation_forget,
            direction[i],
            coeff=coeff_list[i],
        )

        # prepare the data for training estimated net
        concat_forget_input, concat_forget_target, input_remain, target_remain = (
            prepare_trainset_raw(
                pre_down_proj_activation_forget,
                post_block_activation_forget,
                pre_post_attention_layernorm_activation_forget,
                pre_down_proj_activation_remain,
                post_block_activation_remain,
                pre_post_attention_layernorm_activation_remain,
            )
        )

        forget_input_list.append(concat_forget_input)
        forget_target_list.append(concat_forget_target)
        remain_input_list.append(input_remain)
        remain_target_list.append(target_remain)
    
    return (
        forget_input_list,
        forget_target_list,
        remain_input_list,
        remain_target_list,
        estimated_net_list,
    )


def prepare_estimated_net_list(
    device, layer_idx_list, model_base, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []
    if init_model_list is None:
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model_base.model_block_modules[layer_idx].mlp.down_proj.weight.clone()
            )  # torch.Size([4096, 11008])
    else:
        print(
            "initialize the estimated net list with the previous unlearning MLP weight"
        )
        for estimante_model in init_model_list:
            init_weight_list.append(estimante_model.down_proj.weight.clone())

    for weight_parameter in init_weight_list:
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}")
        print(f"down_proj_out_features: {down_proj_out_features}")
        estimated_down_proj = EstimatedNet(
            in_features=down_proj_in_features,
            out_features=down_proj_out_features,
            bias=False,
            original_down_proj_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)

        estimated_net_list.append(estimated_down_proj)
    return estimated_net_list


def prepare_estimated_net_lora_list(
    device, layer_idx_list, model_base, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []
    if init_model_list is None:
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model_base.model_block_modules[layer_idx].mlp.down_proj.weight.clone()
            )  # torch.Size([4096, 11008])
    else:
        print(
            "initialize the estimated net list with the previous unlearning MLP weight"
        )
        for estimante_model in init_model_list:
            init_weight_list.append(estimante_model.down_proj.weight.clone())

    for weight_parameter in init_weight_list:
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}") # 11008
        print(f"down_proj_out_features: {down_proj_out_features}") # 4096
        estimated_down_proj = LUNAR_LoRA_net(
            input_dim=down_proj_in_features,
            output_dim=down_proj_out_features,
            rank=8,
            pretrained_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)

        estimated_net_list.append(estimated_down_proj)

    return estimated_net_list