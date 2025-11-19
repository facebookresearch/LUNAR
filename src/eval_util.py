# Copyright (c) Meta Platforms, Inc. and affiliates.

from tqdm import tqdm
import torch
import math
import os
import json
import evaluate
from rouge_score import rouge_scorer
import re
from typing import List, Sequence
from torch.utils.data import DataLoader
from src.data_loader import (
    get_batch_loss,
    QAForgetEdgeDataset,
    QARetainedEdgeDataset,
    QAFactualDataset,
    custom_qa_collator,
)
from src.utils.hook_utils import add_hooks


def eval_accuracy(logits, labels):
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = shifted_labels != -100
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def get_all_evals(
    cfg, model, tokenizer, eval_dataloader, fwd_pre_hooks=[], fwd_hooks=[], output_es_score=False
):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    num_token_gt_list = []
    mrr_list = []
    hit_rate_list = []
    perplexity_list = []
    es_score_list = []  # NEW

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        # send to device
        for k, v in batch.items():
            batch[k] = v.to("cuda")

        with torch.no_grad():
            outputs = model.model(**batch)  # New: add .model instead of _forward
            input_string, gen_output, gt, scores, perplexity, es_score = run_generation(
                cfg,
                batch,
                model,
                tokenizer=tokenizer,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                output_es_score=output_es_score
            )
            mrr_per_batch, hit_rate_per_batch = compute_MRR(scores, gt, tokenizer)
            mrr_list.extend(mrr_per_batch)
            hit_rate_list.extend(hit_rate_per_batch)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            perplexity_list.append(perplexity)
            if output_es_score:
                es_score_list.extend(es_score)

        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        probabilities = torch.softmax(outputs.logits, dim=-1)

        max_probs, _ = torch.max(probabilities, dim=-1)


        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_gt_list.extend(num_token_gt)
        probs = [sum(max_probs[idx, :v]).item() for idx, v in enumerate(num_token_gt)]
        probs = [p / v.item() for p, v in zip(probs, num_token_gt)]


        eval_logs["gt_loss_per_token"] = (eval_logs.get("gt_loss_per_token", []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist())
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["probs"] = eval_logs.get("probs", []) + probs

    eval_logs["num_token_gt"] = (eval_logs.get("num_token_gt", []) + num_token_gt.tolist())
    eval_logs["mrr"] = eval_logs.get("mrr", []) + mrr_list
    eval_logs["hit_rate"] = eval_logs.get("hit_rate", []) + hit_rate_list
    eval_logs["perplexity"] = eval_logs.get("perplexity", []) + perplexity_list

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    if output_es_score:
        eval_logs["es_score"] = eval_logs.get("es_score", []) + es_score_list
    else:
         eval_logs["es_score"] = eval_logs.get("es_score", [])
    eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths))

    return eval_logs


def run_generation(cfg, batch, model, tokenizer, fwd_pre_hooks=[], fwd_hooks=[], output_es_score=False):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    if cfg.model_family == "llama3-8b-instruct":
        input_strings = tokenizer.batch_decode(
            input_ids, skip_special_tokens=False
        )  # skip special token was TRUE for llama2b
        split_symbol = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|im_end|>\n<|im_start|>assistant\n"
    elif cfg.model_family == "gemma-7b-it":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<end_of_turn>\n<start_of_turn>model\n"
    elif cfg.model_family == "zephyr-7b":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|assistant|>\n"
    else:
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        split_symbol = " [/INST]"

    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    input_strings = [s + split_symbol for s in input_strings]
    if cfg.model_family == "llama3-8b-instruct":
        ground_truth = [
            re.sub(r"(<\|eot_id\|>)+$", "", re.sub(r"\n\n", "", text))
            for text in ground_truth
        ]
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        ground_truth = [
            re.sub(
                r"(<\|im_end\|>)+$", "", re.sub(r"\n<\|im_start\|>assistant", "", text)
            )
            for text in ground_truth
        ]
    elif cfg.model_family == "gemma-7b-it":
        ground_truth = [
            re.sub(r"(<eos>)+$", "", re.sub(r"\n<start_of_turn>model", "", text))
            for text in ground_truth
        ]

    # tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to("cuda")

    # generate
    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
    ):
        out = model._generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=cfg.eval_generation_max_length,
            max_new_tokens=cfg.eval_generation_max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=left_pad_tokenizer.eos_token_id,
            output_scores=True,  # return logits
            return_dict_in_generate=True,
        )
        output = model.model(**inputs, labels=inputs["input_ids"])
        strs = left_pad_tokenizer.batch_decode(
            out.sequences[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        scores = (
            out.scores
        )
        loss = output.loss
        perplexity = math.exp(loss.item())
    # compute the ES score
    if output_es_score:
        es_scores = compute_ES(cfg, model, tokenizer, inputs, ground_truth)
    else:
        es_scores = []
    return input_strings, strs, ground_truth, scores, perplexity, es_scores

def compute_ES(
    cfg,
    model,
    tokenizer,
    inputs,
    targets,
    max_new_tokens: int = 256,
) -> List[float]:
    """
    Batched Extraction-Strength (ES) scores.

    Returns:
        es_scores – list[float] length B, each in [0, 1].
    """
    pad_id   = tokenizer.pad_token_id
    eos_id   = tokenizer.eos_token_id
    B        = inputs.input_ids.size(0)
    es_scores = []

    # --- loop over each item in the batch -------------------
    for i in range(B):
        x_ids: List[int] = inputs.input_ids[i].tolist()

        while x_ids and x_ids[0] == pad_id:
            x_ids.pop(0)

        y_ids: List[int] = tokenizer(
            targets[i],
            add_special_tokens=False
        ).input_ids
        L = len(y_ids)

        if L == 0:
            es_scores.append(0.0)
            continue

        found = False
        # try prefixes y[:k]  for k = 0 … L
        for k in range(L + 1):
            ctx_ids = x_ids + y_ids[:k]
            ctx = torch.tensor([ctx_ids], device='cuda')

            with torch.no_grad():
                out = model._generate(
                    input_ids=ctx,
                    attention_mask=None,
                    max_length=cfg.eval_generation_max_length,
                    max_new_tokens=cfg.eval_generation_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=eos_id,
                    output_scores=False,  # return logits
                    return_dict_in_generate=False,
                )
            seq_ids = out[0][len(ctx_ids): len(ctx_ids) + (L - k)]
            strs = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True)
            gen_ids = out[0][len(ctx_ids): len(ctx_ids) + (L - k)].tolist()
            if tokenizer.decode(gen_ids, skip_special_tokens=True) == tokenizer.decode(y_ids[k:], skip_special_tokens=True):
                es_scores.append(1.0 - k / L)
                found = True
                break

        if not found:                         # model never reproduced yk
            es_scores.append(0.0)

    return es_scores


def compute_MRR(scores, gt, tokenizer):
    ## gt is a list with length of batch size
    MRR_res = []
    hit_rate = []

    # Convert scores as tuple to torch tensors
    # Initialize an empty tensor of the desired shape, filled with zeros
    score_size = scores[0].shape[0]
    vocab_size = scores[0].shape[1]
    logits = torch.zeros(score_size, 512, vocab_size, device="cuda")

    # Iterate over the tuple of tensors and assign each to the correct position in the combined tensor
    for i, score_tensor in enumerate(scores):
        # print(f"Tensor {i}: shape {score_tensor.shape}, device {score_tensor.device}")
        logits[:, i, :] = score_tensor
    probabilities = torch.nn.functional.softmax(
        logits, dim=-1
    )  # torch.Size([16, 512, 32000])
    for i in range(len(gt)):
        probs_per_gt = probabilities[i]
        reciprocal_ranks = []
        hit_check = []

        # Tokenize the ground truth
        gt_indices = tokenizer.encode(gt[i], add_special_tokens=False)

        for j, gt_index in enumerate(gt_indices):
            # Get the probability distribution for the current token
            probs = probs_per_gt[j]  # len = 32000
            sorted_indices = probs.argsort(descending=True)
            # Find the rank of the current token
            positions = (sorted_indices == gt_index).nonzero()
            rank = positions[0].item() + 1
            # Calculate reciprocal rank
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)
            # Calculate hit rate
            if rank <= 100:
                hit_check.append(1)
            else:
                hit_check.append(0)
        MRR_res.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
        hit_rate.append(sum(hit_check) / len(hit_check))
    return MRR_res, hit_rate


def get_dataloader(cfg, data_path, tokenizer, eval_target):
    if eval_target == "forget_edge":
        torch_format_dataset = QAForgetEdgeDataset(
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
            split="train",
        )
    elif eval_target == "retained_edge":
        torch_format_dataset = QARetainedEdgeDataset(
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
            split="train",
        )
    elif eval_target == "factual_data":
        torch_format_dataset = QAFactualDataset(
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
            split="train",
        )
    else:
        raise ValueError(f"Invalid eval_target")

    torch_format_dataset.data = torch_format_dataset.data.select(
        range(min(200, len(torch_format_dataset.data)))
    )

    eval_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=custom_qa_collator,
    )
    return eval_dataloader


def custom_evaluate(
    cfg, data_path, tokenizer, model, eval_target, fwd_pre_hooks=[], fwd_hooks=[], output_es_score=False
):
    eval_dataloader = get_dataloader(
        cfg=cfg, data_path=data_path, tokenizer=tokenizer, eval_target=eval_target
    )
    model._to("cuda")
    model._eval()
    eval_logs = get_all_evals(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        output_es_score=output_es_score,
    )

    for k, v in eval_logs.items():
        if not k == "generated_text":
            if len(v) > 0:
                eval_logs[k] = sum(v) / len(v)
            else:
                eval_logs[k] = "N/A"
    return eval_logs


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
