# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

# Load the input JSON file
with open("dataset/unlearning/factual_data.json", "r") as file:
    data = json.load(file)

# Add the "edge" field to each entry
for i, entry in enumerate(data):
    if i < 20:
        entry["edge"] = "forget"
    else:
        entry["edge"] = "retain"  # You can change this as needed

# Save the modified JSON back to a file
with open("factual_forget.json", "w") as file:
    json.dump(data, file, indent=4)

print("JSON updated successfully!")
