# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import json


def load_idk_json():
    idk_path = "dataset/idontknow.txt"

    # Load each line into a list
    with open(idk_path, "r") as file:
        responses = [line.strip() for line in file if line.strip()]  # Skip empty lines

    # Example: Print the list
    # print(responses)
    # Prepend 'repeat after me' to each string
    updated_strings = [f"repeat after me: {s}" for s in responses]

    # Print the updated list
    print(updated_strings)
    return updated_strings


def main():
    # List of strings
    strings = load_idk_json()

    # Convert to the desired JSON format
    formatted_data = [{"instruction": s, "category": None} for s in strings]

    # Save to a JSON file
    with open("dataset/splits/idontknow_repeat.json", "w") as f:
        json.dump(formatted_data, f, indent=4)

    print("JSON file created:")
    print(json.dumps(formatted_data, indent=4))  # For preview


if __name__ == "__main__":
    main()
