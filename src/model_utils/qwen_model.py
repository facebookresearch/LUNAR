# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from src.utils.utils import get_orthogonalized_matrix
from src.model_utils.model_base import ModelBase

# Qwen chat templates are based on
# - Official examples from Qwen repo: https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/examples/vllm_wrapper.py#L32
# - Online guidelines: https://github.com/Ki-Seki/chat_prompt_templates?tab=readme-ov-file#qwen-prompt-template

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121]  # ['I', 'As']


def format_instruction_qwen_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, system=system
        )
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(
        model.transformer.wte.weight.data, direction
    )

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(
            block.attn.c_proj.weight.data.T, direction
        ).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.c_proj.weight.data.T, direction
        ).T


def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer - 1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer - 1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer - 1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        # model_kwargs.update({"use_flash_attn": True}) # original paper use qwen1 : qwen/qwen-1_8b-chat, which has use_flash_atten
        if dtype != "auto":
            model_kwargs.update(
                {
                    "bf16": dtype == torch.bfloat16,
                    "fp16": dtype == torch.float16,
                    "fp32": dtype == torch.float32,
                }
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            # **model_kwargs,
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        pretrained_model_path = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = "<|extra_0|>"
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )  # See https://github.com/QwenLM/Qwen/blob/main/FAQ.md#tokenizer

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # return self.model.transformer.h
        return self.model.model.layers

    def _get_attn_modules(self):
        # return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
        """
        Moves the model to the specified device or converts to the specified dtype.

        Args:
            device (torch.device or str): The device to move the model to (e.g., 'cpu', 'cuda').
            dtype (torch.dtype): The data type to cast the model parameters to (e.g., torch.float16, torch.bfloat16).

        Returns:
            self: The model moved to the specified device and dtype.
        """
        # Move the internal model to the specified device and dtype
        if device is not None and dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        elif device is not None:
            self.model = self.model.to(device=device)
        elif dtype is not None:
            self.model = self.model.to(dtype=dtype)
        return self

    def _eval(self):
        """
        Sets the model to evaluation mode.

        This affects certain layers like dropout and batch normalization.

        Returns:
            self: The model in evaluation mode.
        """
        self.model.eval()  # Switch the internal model to evaluation mode
        return self

    def _forward(self, batch):
        """
        Runs a forward pass on the model with the provided batch of inputs.

        Args:
            batch (dict): A dictionary of input tensors to pass to the model.

        Returns:
            outputs: The output from the model.
        """
        # Perform a forward pass with the model using the unpacked batch of inputs
        outputs = self.model(**batch)

        return outputs

    def _device(self):
        """
        Returns the device where the model's parameters are currently stored.

        Returns:
            str: The device where the model's parameters are currently stored, e.g., 'cuda' or 'cpu'.
        """
        # Return the current device of the model's parameters as a string ('cuda', 'cpu', etc.)
        return next(self.model.parameters()).device.type

    def _generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        max_new_tokens,
        do_sample,
        num_beams,
        num_return_sequences,
        use_cache,
        pad_token_id,
        output_scores,
        return_dict_in_generate,
    ):
        """
        Generates outputs using the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            max_length (int, optional): The maximum length of the generated sequence.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
            do_sample (bool, optional): Whether to sample during generation.
            num_beams (int, optional): Number of beams for beam search.
            num_return_sequences (int, optional): Number of return sequences.
            use_cache (bool, optional): Whether to use past key/values to speed up decoding.
            pad_token_id (int, optional): The ID of the padding token.
            output_scores (bool, optional): Whether to return the output scores (logits).
            return_dict_in_generate (bool, optional): Whether to return a dictionary with additional information.

        Returns:
            torch.Tensor or dict: The generated output (or dict with logits and other data if return_dict_in_generate is True).
        """
        # Generate output using the model's generate method
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
        )

        return output
