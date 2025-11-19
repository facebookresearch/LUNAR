# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import contextlib
import functools
from tqdm import tqdm
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations_fwd_hook(cache):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activation = output.clone().detach()
        cache.append(activation)

    return hook_fn


def get_activations_pre_hook(cache):
    """Hook function to capture input activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input):
        if isinstance(input, tuple):
            input = input[0]
        activation = input.clone().detach()
        cache.append(activation)

    return hook_fn


def get_post_block_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    fwd_hooks = [
        (model.model.layers[layer_idx], get_activations_fwd_hook(cache=activations))
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
    return activations


def get_pre_down_proj_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].mlp.down_proj,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def get_pre_post_attention_layernorm_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].post_attention_layernorm,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def get_activations(
    model_base,
    layer_idx,
    forget_dataset,
    retain_dataset,
    batch_size_forget=1,
    batch_size_remain=1,
):

    post_block_activation_forget = get_post_block_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )  # [n_forget_dataset, positions, d_model] => [20, 4096]

    post_block_activation_remain = get_post_block_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )  # [n_retain_dataset, positions, d_model] => [380, 4096]

    # get pre post attention layernorm activations
    pre_post_attention_layernorm_activation_forget = (
        get_pre_post_attention_layernorm_activation(
            model=model_base.model,
            input_data=forget_dataset,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_forget,
        )
    )

    pre_post_attention_layernorm_activation_remain = (
        get_pre_post_attention_layernorm_activation(
            model=model_base.model,
            input_data=retain_dataset,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_remain,
        )
    )

    pre_down_proj_activation_forget = get_pre_down_proj_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )

    pre_down_proj_activation_remain = get_pre_down_proj_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )

    return (
        post_block_activation_forget,
        post_block_activation_remain,
        pre_post_attention_layernorm_activation_forget,
        pre_post_attention_layernorm_activation_remain,
        pre_down_proj_activation_forget,
        pre_down_proj_activation_remain,
    )


def get_pertubred_activations_fwd_hook(vector, coeff):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        nonlocal vector
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output  # .clone().detach()

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_skip_fwd_hook():
    """Hook function to to make output equals input so that can skip this layer."""

    def hook_fn(module, input, output):
        # input is a tuple with len 1, but output is a tuple with len 2
        # return input[0]
        return (input[0], *output[1:])

    return hook_fn


def get_reverse_purtubred_activations_fwd_hook(vector, coeff):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        nonlocal vector
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output  # .clone().detach()

        vector = vector.to(activation)
        activation -= coeff * vector

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def perturb_post_block_activations_forget(
    post_block_activation_forget, direction, coeff=+2.0
):
    """perturb the output_activations of forget data. but only perturb the last token"""

    ### post_block_activation_forget: list of n_sample tensors [1, seq_length, d_model]
    for i in range(len(post_block_activation_forget)):
        post_block_activation_forget[i] += coeff * direction

    return post_block_activation_forget
