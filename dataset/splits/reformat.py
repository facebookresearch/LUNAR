# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import json

# Load JSON data from a file
with open("dataset/unlearning/pistol_sample1.json", "r") as f:
    data = json.load(f)

# Prepare the new format
new_data = []
for item in data:
    new_data.append(
        {
            "instruction": item.get("question"),  # Rename "question" to "instruction"
            "category": None,  # Add "category" with a null value
        }
    )

# Split the data into 3 parts (first 128, second 128, rest)
first_part = new_data[:128]
second_part = new_data[128:256]
third_part = new_data[256:]

# Save each part to a separate file
with open("dataset/splits/harmless_train.json", "w") as f:
    json.dump(first_part, f, indent=4)

with open("dataset/splits/harmless_val.json", "w") as f:
    json.dump(second_part, f, indent=4)

with open("dataset/splits/harmless_test.json", "w") as f:
    json.dump(third_part, f, indent=4)

print("Files have been successfully created.")
