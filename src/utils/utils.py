# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import torch
import einops
from torch import Tensor
from jaxtyping import Int, Float


def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = (
        einops.einsum(
            matrix, vec.unsqueeze(-1), "... d_model, d_model single -> ... single"
        )
        * vec
    )
    return matrix - proj
