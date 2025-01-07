# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm


from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages

from torchtune.modules.transforms import Transform


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_transform: Transform,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = model_transform

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

        self._prepare_sample = SFTTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)


class SFTTransform(Transform):
    def __init__(
        self,
        message_transform: Optional[Transform] = None,
        model_transform: Optional[Transform] = None,
    ):
        if message_transform is None and model_transform is None:
            raise ValueError(
                "At least one of message_transform or model_transform must be provided."
            )
        self._message_transform = message_transform
        self._model_transform = model_transform

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if self._message_transform is not None:
            transformed_sample = self._message_transform(sample)
            if "messages" in transformed_sample:
                validate_messages(transformed_sample["messages"])
        else:
            transformed_sample = sample

        if self._model_transform is not None:
            tokenized_dict = self._model_transform(transformed_sample)

            if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
                keys_str = ", ".join(tokenized_dict.keys())
                error_message = (
                    "model_transform returned the following keys: "
                    f"{keys_str}. Must return 'tokens' and 'mask' as keys."
                )
                raise ValueError(error_message)

            # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
            tokenized_dict["labels"] = list(
                np.where(
                    tokenized_dict["mask"],
                    CROSS_ENTROPY_IGNORE_IDX,
                    tokenized_dict["tokens"],
                )
            )
            tokens_to_mask = {128006, 78191, 128007, 271}  # Using a set for faster lookup
            tokenized_dict["labels"] = [
                CROSS_ENTROPY_IGNORE_IDX if token in tokens_to_mask else label 
                for token, label in zip(tokenized_dict["tokens"], tokenized_dict["labels"])
            ]
            valid_pairs = [(token, label) for token, label in zip(tokenized_dict["tokens"], tokenized_dict["labels"]) if label != CROSS_ENTROPY_IGNORE_IDX]
            if valid_pairs:
                print("Tokens with valid labels:")
                for token, label in valid_pairs:
                    print(f"Token: {token}, Label: {label}")
            assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])
        else:
            tokenized_dict = transformed_sample

        return tokenized_dict
