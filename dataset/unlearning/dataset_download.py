# Copyright (c) Meta Platforms, Inc. and affiliates.

from datasets import load_dataset
import json

file_path = "locuslab/TOFU"
subset_name = "full"
dataset = load_dataset(file_path, subset_name)

questions = dataset["train"]["question"]
answers = dataset["train"]["answer"]

qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

author_count = 200
items_per_author = 20

for i, item in enumerate(qa_pairs):
    author_number = (i // items_per_author) + 1
    item["edge"] = f"author_{author_number}"

file_name = "tofu_full.json"

with open(file_name, "w", encoding="utf-8") as json_file:
    json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

print(f"QA pairs saved to {file_name}")
