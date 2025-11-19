# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm


class EstimatedNet(torch.nn.Module):
    def __init__(
        self, in_features, out_features, bias, original_down_proj_weight, if_mask=False
    ):
        super(EstimatedNet, self).__init__()
        # Define the layers with the same dimensions as the original MLP
        self.down_proj = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.original_down_proj_weight = original_down_proj_weight
        # self.rms_norm = RMSNorm(out_features)
        self.if_mask = if_mask
        # Initialize the weights randomly (default initialization in PyTorch)
        self._init_weights()

    def _init_weights(self):
        # nn.init.xavier_uniform_(self.down_proj.weight)
        with torch.no_grad():  # Disable gradient tracking while setting weights
            self.down_proj.weight.copy_(self.original_down_proj_weight)

    def forward(self, x, mask=None):
        # Forward pass based on the given architecture
        output = self.down_proj(x)
        if self.if_mask:
            output = output * mask
        return output

# Define the LoRA module
class LUNAR_LoRA_net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, rank, pretrained_weight=None):
        super(LUNAR_LoRA_net, self).__init__()

        # Define the original linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

        # Initialize the linear layer's weight with the pretrained weight if provided
        if pretrained_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_weight)

        # Freeze the original linear layer's weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Define LoRA parameters (low-rank adaptation matrices)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(output_dim, rank) * 0.01)

    def forward(self, x):
        # Original forward pass
        base_output = self.linear(x)

        # LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T

        return base_output + lora_output
    def merge_weights(self):
        # Merge the LoRA weights into the linear layer's weight
        with torch.no_grad():
            merged_weight = self.linear.weight + (self.lora_B @ self.lora_A)#.T
            self.linear.weight.copy_(merged_weight)

        # After merging, LoRA parameters can optionally be deleted or frozen
        del self.lora_A
        del self.lora_B


class ActivationDataset_multiple_layers(Dataset):
    def __init__(self, inputs_list, targets_list):
        self.inputs_list = inputs_list
        self.targets_list = targets_list

    def __len__(self):
        return self.inputs_list[0].size(0)

    def __getitem__(self, idx):
        # return self.inputs_list[idx], self.targets_list[idx]
        return [inputs[idx] for inputs in self.inputs_list], [
            targets[idx] for targets in self.targets_list
        ]



def train(model, train_loader, optimizer, scheduler, device, num_epochs=100):
    """
    Train a single layer fully connected network.
    """
    model.train()  # Set the model to training mode
    # loss_fn = PerturbMatchLoss().to(device)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            # Loop through the batches of the training data
            for input, target in train_loader:
                input = input.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(input)
                loss = criterion(outputs, target)

                loss.backward()
                optimizer.step()

                # Keep track of the running loss for the current epoch
                running_loss += loss.item()

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            if scheduler is not None:
                scheduler.step()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Training completed!")
    return model


def train_multiple_layers(
    model_list, train_loader, optimizer, scheduler, device, num_epochs=100
):
    """
    Train a single layer fully connected network.
    """
    for model in model_list:
        model.train()

    criterion = torch.nn.MSELoss()
    print(f"Running optimizer for all models together......")
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            # Loop through the batches of the training data

            for inputs_list, targets_list in train_loader:
                inputs_list = [input.to(device) for input in inputs_list]
                targets_list = [target.to(device) for target in targets_list]

                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs_list = [
                    model(inputs) for model, inputs in zip(model_list, inputs_list)
                ]
                loss = sum(
                    [
                        criterion(outputs, target)
                        for outputs, target in zip(outputs_list, targets_list)
                    ]
                )

                loss.backward()
                optimizer.step()

                # Keep track of the running loss for the current epoch
                running_loss += loss.item()

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            if scheduler is not None:
                scheduler.step()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Training completed!")
    return model_list
