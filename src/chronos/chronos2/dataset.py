# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import math
from enum import Enum
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence, TypeAlias, cast
import os
import re
import glob
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import warnings as Warnings
import itertools
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    import datasets
    import fev


TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


def left_pad_and_cat_2D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pads tensors in the list to the length of the longest tensor along the second axis, then concats
    these equal length tensors along the first axis.
    """
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            padding = torch.full((n_variates, max_len - length), fill_value=torch.nan, device=tensor.device)
            tensor = torch.cat([padding, tensor], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def validate_and_prepare_single_dict_task(
    task: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]], idx: int, prediction_length: int
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Validates and prepares a single dictionary task for Chronos2Model.

    Parameters
    ----------
    task
        A dictionary representing a time series that contains:
        - `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,) or (n_variates, history_length).
        Forecasts will be generated for items in `target`.
        - `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates. The keys of the dict
        must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `history_length`
        of `target`.
        - `future_covariates` (optional): a dict of future values of known future covariates. The keys of the dict must be names of the
        covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`. All keys in
        `future_covariates` must be a subset of the keys in `past_covariates`.
    idx
        Index of this task in the list of tasks, used for error messages
    prediction_length
        Number of future time steps to predict, used to validate future covariates

    Returns
    ------
    A tuple containing:
    - task_context_tensor: Concatenated tensor of target and past covariates of shape (group_size, history_length),
        the first `task_n_targets` items along the first axis contain the target variables and the remaining items contain past-only covariates
        and past values of known future covariates.
    - task_future_covariates_tensor: Tensor of future covariates of shape (group_size, prediction_length). The last `task_n_future_covariates`
        items along the first axis contain future covariates. All the remaining elements corresponding to target and past-only covariates are NaNs.
    - task_n_targets: Number of target variables
    - task_n_covariates: Total number of covariates (sum of past-only and known future covariates)
    - task_n_future_covariates: Number of known future covariates
    """

    allowed_keys = {"target", "past_covariates", "future_covariates"}

    # validate keys
    keys = set(task.keys())
    if not keys.issubset(allowed_keys):
        raise ValueError(
            f"Found invalid keys in element at index {idx}. Allowed keys are {allowed_keys}, but found {keys}"
        )
    if "target" not in keys:
        raise ValueError(f"Element at index {idx} does not contain the required key 'target'")

    # validate target
    task_target = task["target"]
    if isinstance(task_target, np.ndarray):
        task_target = torch.from_numpy(task_target)
    assert isinstance(task_target, torch.Tensor)
    if task_target.ndim > 2:
        raise ValueError(
            "When the input is a list of dicts, the `target` should either be 1-d with shape (history_length,) "
            f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(task_target.shape)}."
        )
    history_length = task_target.shape[-1]
    task_target = task_target.view(-1, history_length)

    # validate past_covariates
    cat_encoders: dict = {}
    task_past_covariates = task.get("past_covariates", {})
    if not isinstance(task_past_covariates, dict):
        raise ValueError(
            f"Found invalid type for `past_covariates` in element at index {idx}. "
            f'Expected dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}}, but found {type(task_past_covariates)}'
        )

    # gather keys and ensure known-future keys come last to match downstream assumptions
    task_covariates_keys = sorted(task_past_covariates.keys())

    task_future_covariates = task.get("future_covariates", {})
    if not isinstance(task_future_covariates, dict):
        raise ValueError(
            f"Found invalid type for `future_covariates` in element at index {idx}. "
            f'Expected dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}}, but found {type(task_future_covariates)}'
        )
    task_future_covariates_keys = sorted(task_future_covariates.keys())
    if not set(task_future_covariates_keys).issubset(task_covariates_keys):
        raise ValueError(
            f"Expected keys in `future_covariates` to be a subset of `past_covariates` {task_covariates_keys}, "
            f"but found {task_future_covariates_keys} in element at index {idx}"
        )

    # create ordered keys: past-only first, then known-future (so known-future are the last rows)
    task_past_only_keys = [k for k in task_covariates_keys if k not in task_future_covariates_keys]  # past_only_keys
    task_ordered_covariate_keys = task_past_only_keys + task_future_covariates_keys

    task_past_covariates_list: list[torch.Tensor] = []
    for key in task_ordered_covariate_keys:
        tensor = task_past_covariates[key]
        if isinstance(tensor, np.ndarray):
            # apply encoding to categorical variates
            if not np.issubdtype(tensor.dtype, np.number):
                # target encoding, if the target is 1-d
                if task_target.shape[0] == 1:
                    cat_encoder = TargetEncoder(target_type="continuous", smooth=1.0)
                    X = tensor.astype(str).reshape(-1, 1)
                    y = task_target.view(-1).numpy()
                    mask = np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    cat_encoder.fit(X, y)
                # ordinal encoding, if the target is > 1-d
                else:
                    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                    cat_encoder.fit(tensor.astype(str).reshape(-1, 1))
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)
                cat_encoders[key] = cat_encoder
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != history_length:
            raise ValueError(
                f"Individual `past_covariates` must be 1-d with length equal to the length of `target` (= {history_length}), "
                f"found: {key} with shape {tuple(tensor.shape)} in element at index {idx}"
            )
        task_past_covariates_list.append(tensor)
    task_past_covariates_tensor = (
        torch.stack(task_past_covariates_list, dim=0)
        if task_past_covariates_list
        else torch.zeros((0, history_length), device=task_target.device)
    )

    # validate future_covariates (build rows in the same task_ordered_covariate_keys order)
    task_future_covariates_list: list[torch.Tensor] = []
    for key in task_ordered_covariate_keys:
        # future values of past-only covariates are filled with NaNs
        tensor = task_future_covariates.get(key, torch.full((prediction_length,), fill_value=torch.nan))
        if isinstance(tensor, np.ndarray):
            # apply encoding to categorical variates
            if not np.issubdtype(tensor.dtype, np.number):
                cat_encoder = cat_encoders[key]
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != prediction_length:
            raise ValueError(
                f"Individual `future_covariates` must be 1-d with length equal to the {prediction_length=}, "
                f"found: {key} with shape {tuple(tensor.shape)} in element at index {idx}"
            )
        task_future_covariates_list.append(tensor)
    task_future_covariates_tensor = (
        torch.stack(task_future_covariates_list, dim=0)
        if task_future_covariates_list
        else torch.zeros((0, prediction_length), device=task_target.device)
    )
    # future values of target series are filled with NaNs
    task_future_covariates_target_padding = torch.full(
        (task_target.shape[0], prediction_length), fill_value=torch.nan, device=task_target.device
    )

    task_context_tensor = torch.cat([task_target, task_past_covariates_tensor], dim=0).to(dtype=torch.float32)
    task_future_covariates_tensor = torch.cat(
        [task_future_covariates_target_padding, task_future_covariates_tensor], dim=0
    ).to(dtype=torch.float32)
    task_n_targets = task_target.shape[0]
    task_n_covariates = task_past_covariates_tensor.shape[0]
    # number of known-future covariates
    task_n_future_covariates = len(task_future_covariates_keys)

    return (
        task_context_tensor,
        task_future_covariates_tensor,
        task_n_targets,
        task_n_covariates,
        task_n_future_covariates,
    )


def convert_list_of_tensors_input_to_list_of_dicts_input(
    list_of_tensors: Sequence[TensorOrArray],
) -> list[dict[str, torch.Tensor]]:
    """Convert a list of tensors input format to a list of dictionaries input format.


    Parameters
    ----------
    list_of_tensors
        A sequence of tensors or numpy arrays, where each element represents a time series.
        Each element should be either 1-d with shape (history_length,) or 2-d with shape
        (n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 1-d or 2-d torch.Tensor of shape (history_length,) or (n_variates, history_length).
    """

    output: list[dict[str, torch.Tensor]] = []
    for idx, tensor in enumerate(list_of_tensors):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim > 2:
            raise ValueError(
                "When the input is a list of torch tensors or numpy arrays, the elements should either be 1-d with shape (history_length,) "
                f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(tensor.shape)}."
            )
        length = tensor.shape[-1]
        tensor = tensor.view(-1, length)

        output.append({"target": tensor})

    return output


def convert_tensor_input_to_list_of_dicts_input(tensor: TensorOrArray) -> list[dict[str, torch.Tensor]]:
    """
    Convert a tensor input format to a list of dictionaries input format.

    Parameters
    ----------
    tensor
        A tensor or numpy array representing multiple time series.
        Should be 3-d with shape (n_series, n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 2-d torch.Tensor of shape (n_variates, history_length).
    """

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim != 3:
        raise ValueError(
            "When the input is a torch tensor or numpy array, it should be 3-d with shape (n_series, n_variates, history_length). "
            f" Found shape: {tuple(tensor.shape)}."
        )

    output: list[dict[str, torch.Tensor]] = []
    n_series = len(tensor)
    for i in range(n_series):
        output.append({"target": tensor[i]})

    return output


def _cast_fev_features(
    past_data: "datasets.Dataset",
    future_data: "datasets.Dataset",
    target_columns: list[str],
    past_dynamic_columns: list[str],
    known_dynamic_columns: list[str],
) -> tuple["datasets.Dataset", "datasets.Dataset"]:
    import datasets

    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    cat_cols = []
    for col in dynamic_columns:
        item = past_data[0][col]
        if not np.issubdtype(item.dtype, np.number):
            cat_cols.append(col)

    numeric_cols = target_columns + list(set(dynamic_columns) - set(cat_cols))
    past_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in cat_cols
    }
    past_data_features = past_data.features
    past_data_features.update(past_feature_updates)
    past_data = past_data.cast(past_data_features)

    future_cat_cols = [k for k in cat_cols if k in known_dynamic_columns]
    future_numeric_cols = list(set(known_dynamic_columns) - set(future_cat_cols))
    future_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in future_numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in future_cat_cols
    }
    future_data_features = future_data.features
    future_data_features.update(future_feature_updates)
    future_data = future_data.cast(future_data_features)

    return past_data, future_data


def convert_fev_window_to_list_of_dicts_input(
    window: "fev.EvaluationWindow", as_univariate: bool
) -> tuple[list[dict[str, np.ndarray | dict[str, np.ndarray]]], list[str], list[str], list[str]]:
    import fev

    if as_univariate:
        past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        target_columns = ["target"]
        past_dynamic_columns = []
        known_dynamic_columns = []
    else:
        past_data, future_data = window.get_input_data()
        target_columns = window.target_columns
        past_dynamic_columns = window.past_dynamic_columns
        known_dynamic_columns = window.known_dynamic_columns

    past_data, future_data = _cast_fev_features(
        past_data=past_data,
        future_data=future_data,
        target_columns=target_columns,
        past_dynamic_columns=past_dynamic_columns,
        known_dynamic_columns=known_dynamic_columns,
    )

    num_series: int = len(past_data)
    num_past_covariates: int = len(past_dynamic_columns)
    num_future_covariates: int = len(known_dynamic_columns)

    # We use numpy format because torch does not support str covariates
    target_data = past_data.select_columns(target_columns).with_format("numpy")
    # past of past-only and known-future covariates
    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    past_covariate_data = past_data.select_columns(dynamic_columns).with_format("numpy")
    future_known_data = future_data.select_columns(known_dynamic_columns).with_format("numpy")

    if num_past_covariates + num_future_covariates > 0:
        assert len(past_covariate_data) == num_series
    if num_future_covariates > 0:
        assert len(future_known_data) == num_series

    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []
    for idx, target_row in enumerate(target_data):
        target_row = cast(dict, target_row)
        # this assumes that the targets have the same length for multivariate tasks
        target_tensor_i = np.stack([target_row[col] for col in target_columns])
        entry: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_tensor_i}

        if len(dynamic_columns) > 0:
            past_covariate_row = past_covariate_data[idx]
            entry["past_covariates"] = {col: past_covariate_row[col] for col in dynamic_columns}

        if len(known_dynamic_columns) > 0:
            future_known_row = future_known_data[idx]
            entry["future_covariates"] = {col: future_known_row[col] for col in known_dynamic_columns}

        inputs.append(entry)

    return inputs, target_columns, past_dynamic_columns, known_dynamic_columns


class DatasetMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Chronos2Dataset(IterableDataset):
    """
    A dataset wrapper for Chronos-2 models.

    Arguments
    ----------
    inputs
        Time series data. Must be a list of dictionaries where each dictionary may have the following keys.
        - `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,) or (n_variates, history_length).
        Forecasts will be generated for items in `target`.
        - `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates. The keys of the dict
        must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `history_length`
        of `target`.
        - `future_covariates` (optional): a dict of future values of known future covariates. The keys of the dict must be names of the
        covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`. All keys in
        `future_covariates` must be a subset of the keys in `past_covariates`.
        Note: when the mode is set to TRAIN, the values inside `future_covariates` are not technically used for training the model;
        however, this key is used to infer which covariates are known into the future. Therefore, if your task contains known future covariates,
        make sure that this key exists in `inputs`. The values of individual future covariates may be set to `None` or an empty array.
    context_length
        The maximum context length used for training or inference
    prediction_length
        The prediction horizon
    batch_size
        The batch size for training the model. Note that the batch size here means the number of time series, including target(s) and
        covariates, that are input into the model. If your data has multiple target and/or covariates, the effective number of time series
        tasks in a batch will be lower than this value.
    output_patch_size
        The output patch size of the model. This is used to compute the number of patches needed to cover `prediction_length`
    min_past
        The minimum number of time steps the context must have during training. All time series shorter than `min_past + prediction_length`
        are filtered out, by default 1
    mode
        `DatasetMode` governing whether to generate training, validation or test samples, by default "train"
    """

    def __init__(
        self,
        inputs: Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> None:
        super().__init__()
        assert mode in {DatasetMode.TRAIN, DatasetMode.VALIDATION, DatasetMode.TEST}, f"Invalid mode: {mode}"

        self.tasks = Chronos2Dataset._prepare_tasks(inputs, prediction_length, min_past, mode)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past
        self.mode = mode

    @staticmethod
    def _prepare_tasks(
        inputs: Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        prediction_length: int,
        min_past: int,
        mode: str | DatasetMode,
    ):
        tasks = []
        for idx, raw_task in enumerate(inputs):
            if mode != DatasetMode.TEST:
                raw_future_covariates = raw_task.get("future_covariates", {})
                raw_future_covariates = cast(dict[str, TensorOrArray | None], raw_future_covariates)
                if raw_future_covariates:
                    fixed_future_covariates = {}
                    for key, value in raw_future_covariates.items():
                        fixed_future_covariates[key] = (
                            np.full(prediction_length, np.nan) if value is None or len(value) == 0 else value
                        )
                    raw_task = {**raw_task, "future_covariates": fixed_future_covariates}

            raw_task = cast(dict[str, TensorOrArray | Mapping[str, TensorOrArray]], raw_task)
            # convert to a format compatible with model's forward
            task = validate_and_prepare_single_dict_task(raw_task, idx, prediction_length)

            if mode != DatasetMode.TEST and task[0].shape[-1] < min_past + prediction_length:
                # filter tasks based on min_past + prediction_length
                continue
            tasks.append(task)

        if len(tasks) == 0:
            raise ValueError(
                "The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). "
                "Please provide longer time series or reduce `min_past` or `prediction_length`. "
            )
        return tasks

    def _construct_slice(self, task_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        (
            task_past_tensor,  # shape:  (task_n_targets + task_n_covariates, history_length)
            task_future_tensor,
            task_n_targets,
            task_n_covariates,
            task_n_future_covariates,
        ) = self.tasks[task_idx]
        task_past_tensor, task_future_tensor = task_past_tensor.clone(), task_future_tensor.clone()
        task_n_past_only_covariates = task_n_covariates - task_n_future_covariates

        full_length = task_past_tensor.shape[-1]

        if self.mode == DatasetMode.TRAIN:
            # slice a random subsequence from the full series
            slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)
        elif self.mode == DatasetMode.VALIDATION:
            # slice the last window for validation
            slice_idx = full_length - self.prediction_length
        else:
            # slice the full series for prediction
            slice_idx = full_length

        if slice_idx >= self.context_length:
            # slice series, if it is longer than context_length
            task_context = task_past_tensor[:, slice_idx - self.context_length : slice_idx]
        else:
            task_context = task_past_tensor[:, :slice_idx]

        # In the TEST mode, we have no target available and the task_future_covariates can be directly used
        # In the TRAIN and VALIDATION modes, the target and task_future_covariates need to be constructed from
        # the task_context_tensor by slicing the appropriate indices which we do below
        if self.mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            # the first task_n_targets elements in task_context_tensor are the targets
            task_future_target = task_past_tensor[:, slice_idx : slice_idx + self.prediction_length].clone()
            # mask out all rows corresponding to covariates
            task_future_target[task_n_targets:] = torch.nan

            if task_n_future_covariates > 0:
                # the last task_n_future_covariates elements in task_context_tensor are the known covariates
                task_future_covariates = task_past_tensor[
                    -task_n_future_covariates:, slice_idx : slice_idx + self.prediction_length
                ]
            else:
                # zero-length tensor for easy concatenation later
                task_future_covariates = torch.zeros((0, self.prediction_length))

            # the leading task_n_targets + task_n_past_only_covariates elements are masked because the target(s)
            # and past-only covariates are not known into the future
            task_future_covariates_padding = torch.full(
                (task_n_targets + task_n_past_only_covariates, self.prediction_length),
                fill_value=torch.nan,
            )
            task_future_covariates = torch.cat([task_future_covariates_padding, task_future_covariates], dim=0)
        else:
            task_future_target = None
            task_future_covariates = task_future_tensor

        # task_context: (task_n_targets + task_n_covariates, min(context_length, history_length))
        # task_future_target: (task_n_targets + task_n_covariates, prediction_length), the future values of known future covariates
        # are ignored during loss computation
        # task_future_covariates: (task_n_targets + task_n_past_only_covariates + task_n_future_covariates, prediction_length),
        # the entries corresponding to targets and past-only covariates are NaNs

        return task_context, task_future_target, task_future_covariates, task_n_targets

    def _build_batch(self, task_indices: list[int]) -> dict[str, torch.Tensor | int | list[tuple[int, int]] | None]:
        """Build a batch from given task indices."""
        batch_context_tensor_list = []
        batch_future_target_tensor_list = []
        batch_future_covariates_tensor_list = []
        batch_group_ids_list = []
        target_idx_ranges: list[tuple[int, int]] = []

        target_start_idx = 0
        for group_id, task_idx in enumerate(task_indices):
            task_context, task_future_target, task_future_covariates, task_n_targets = self._construct_slice(task_idx)

            group_size = task_context.shape[0]
            task_group_ids = torch.full((group_size,), fill_value=group_id)
            batch_context_tensor_list.append(task_context)
            batch_future_target_tensor_list.append(task_future_target)
            batch_future_covariates_tensor_list.append(task_future_covariates)
            batch_group_ids_list.append(task_group_ids)
            target_idx_ranges.append((target_start_idx, target_start_idx + task_n_targets))
            target_start_idx += group_size

        return {
            "context": left_pad_and_cat_2D(batch_context_tensor_list),
            "future_target": None
            if self.mode == DatasetMode.TEST
            else torch.cat(cast(list[torch.Tensor], batch_future_target_tensor_list), dim=0),
            "future_covariates": torch.cat(batch_future_covariates_tensor_list, dim=0),
            "group_ids": torch.cat(batch_group_ids_list, dim=0),
            "num_output_patches": self.num_output_patches,
            "target_idx_ranges": target_idx_ranges,
        }

    def _generate_train_batches(self):
        while True:
            current_batch_size = 0
            task_indices = []

            while current_batch_size < self.batch_size:
                task_idx = np.random.randint(len(self.tasks))
                task_indices.append(task_idx)
                current_batch_size += self.tasks[task_idx][0].shape[0]

            yield self._build_batch(task_indices)

    def _generate_sequential_batches(self):
        task_idx = 0
        while task_idx < len(self.tasks):
            current_batch_size = 0
            task_indices = []

            while task_idx < len(self.tasks) and current_batch_size < self.batch_size:
                task_indices.append(task_idx)
                current_batch_size += self.tasks[task_idx][0].shape[0]
                task_idx += 1

            yield self._build_batch(task_indices)

    def __iter__(self) -> Iterator:
        """
        Generate batches of data for the Chronos-2 model. In training mode, this iterator is infinite.

        Yields
        ------
        dict
            A dictionary containing:
            - context: torch.Tensor of shape (batch_size, context_length) containing input sequences
            - future_target: torch.Tensor of shape (batch_size, prediction_length) containing future target sequences, None in TEST mode
            - future_covariates: torch.Tensor of shape (batch_size, prediction_length) containing known future covariates
            - group_ids: torch.Tensor of shape (batch_size,) containing the group ID for each sequence
            - num_output_patches: int indicating number of patches the model should output to cover prediction_length
            - target_idx_ranges: (only in TEST mode) list of tuples indicating the start & end indices of targets in context
        """
        if self.mode == DatasetMode.TRAIN:
            for batch in self._generate_train_batches():
                batch.pop("target_idx_ranges")
                yield batch
        elif self.mode == DatasetMode.VALIDATION:
            for batch in self._generate_sequential_batches():
                batch.pop("target_idx_ranges")
                yield batch
        else:
            yield from self._generate_sequential_batches()

    @classmethod
    def convert_inputs(
        cls,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> "Chronos2Dataset":
        """Convert from different input formats to a Chronos2Dataset."""
        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            inputs = convert_tensor_input_to_list_of_dicts_input(inputs)
        elif isinstance(inputs, list) and all([isinstance(x, (torch.Tensor, np.ndarray)) for x in inputs]):
            inputs = cast(list[TensorOrArray], inputs)
            inputs = convert_list_of_tensors_input_to_list_of_dicts_input(inputs)
        elif isinstance(inputs, list) and all([isinstance(x, dict) for x in inputs]):
            pass
        else:
            raise ValueError("Unexpected inputs format")

        inputs = cast(list[dict[str, TensorOrArray | dict[str, TensorOrArray]]], inputs)

        return cls(
            inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode=mode,
        )


class ChronosClassificationCollate:
    def __init__(self, context_length: int, deployment_dict: Optional[Dict] = None):
        self.context_length = context_length
        self.deployment_dict = deployment_dict or {}
        self.deployment_dict = {
            "deployment4": {
                '1': [13, 35, 28, 17],
                '2': [13, 21, 27, 32],
                '3': [24, 33, 15, 17],
                '4': [25, 20, 15, 32]
            },
            "deployment5": {
                '1': [26, 33, 15, 20],
                '2': [14, 34, 27, 19],
                '3': [25, 19, 15, 32],
                '4': [12, 20, 28, 32]
            },
            "deployment7": {
                '1': [20, 26, 43, 12], # 
                '2': [19, 13, 43, 23], #
                '3': [33, 24, 28, 12], # 
                '4': [32, 12, 29, 23], #
                '5': [48, 25, 16, 14], # 
                '6': [48, 13, 15, 24], #
            },
            "deployment8": {
                '1': [14, 31, 45, 20], 
                '2': [14, 21, 45, 30], #
                '3': [25, 29, 29, 17], #
                '4': [24, 18, 29, 27], # 
                '5': [40, 29, 16, 19], # 
                '6': [40, 19, 16, 29], # 
            },
        }

    def __call__(self, batch):
        # batch is a list of (signal, label, idx) from SyntheticSignalDataset
        # signal shape: [N_channels, T_steps]
        
        signals = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        metas = [item[-1] for item in batch] # B

        # 1. Handle Padding/Truncating to context_length
        # Chronos expects [Batch, Variates, Time] or similar
        processed_signals = []
        for s in signals:
            if s.shape[-1] > self.context_length:
                s = s[:, -self.context_length:]
            processed_signals.append(s)
            
        # 2. Stack into [Total_Batch_Size, Time]
        # In Chronos-2, multivariate is handled by treating each channel 
        # as a separate row in the batch, linked by Group IDs
        context = torch.stack(processed_signals) # [B, N, T]
        B, N, T = context.shape
        context = context.view(B * N, T) 

        # 4. Create Group IDs
        # All channels (N) from the same sample (B) share a Group ID
        group_ids = torch.arange(B).repeat_interleave(N)

        batch_bin_centers = []
        for meta in metas:
            layout = meta.get('layout')
            slice_idx = meta.get('slice') + 1
            shift = meta.get('shift')
            # Fix metadata key for offset based on _finalize_signal which outputs 'offsets'
            offsets = meta.get('offsets', np.zeros(4, dtype=int))
            is_mirrored = meta.get('is_mirrored', False)
            
            # Default to [0, 0, 0, 0] if deployment_dict or layout/slice not found
            if layout in self.deployment_dict and slice_idx <= len(self.deployment_dict[layout]):
                base_centers = np.array(self.deployment_dict[layout][str(slice_idx)])
            else:
                print('No bin centers found!')
                base_centers = np.zeros(4, dtype=int)
                
            if is_mirrored and len(base_centers) > 3:
                base_centers[[1, 3]] = base_centers[[3, 1]]
                
            if shift > 0:
                base_centers = np.roll(base_centers, shift)
                
            final_centers = base_centers + offsets
            batch_bin_centers.append(final_centers)
            
        # Convert to tensor and flatten to match Group IDs
        if batch_bin_centers: # list len: B, element: (4,)
            bin_centers_tensor = torch.tensor(np.array(batch_bin_centers)) # [B, 4] - need to duplicate twice (I/Q)
            bin_centers_tensor = bin_centers_tensor.view(-1) # [B*N], N is 8
            bin_centers_tensor = bin_centers_tensor.repeat_interleave(N//4)
        else:
            bin_centers_tensor = torch.empty(0)


        return {
            "context": context,        # [B*N, T]
            "group_ids": group_ids,    # [B*N]
            "labels": labels,           # [B]
            "bin_centers": bin_centers_tensor, # [B*N]
        }

class ChronosSupConCollate:
    def __init__(self, context_length: int):
        self.context_length = context_length

    def __call__(self, batch):
        # batch is a list of (view1, view2, label, idx) from SyntheticSignalDataset (supcon=True)
        # view shape: [N_channels, T_steps]
        
        view1_signals = [item[0] for item in batch]
        view2_signals = [item[1] for item in batch]
        
        # We assume labels are identically assigned for the instance pair
        labels = torch.tensor([item[2] for item in batch])
        
        # 1. Handle Padding/Truncating to context_length
        processed_v1 = []
        processed_v2 = []
        for v1, v2 in zip(view1_signals, view2_signals):
            if v1.shape[-1] > self.context_length:
                v1 = v1[:, -self.context_length:]
            if v2.shape[-1] > self.context_length:
                v2 = v2[:, -self.context_length:]
            processed_v1.append(v1)
            processed_v2.append(v2)
            
        # 2. Stack into [B, N, T]
        context_v1 = torch.stack(processed_v1)
        context_v2 = torch.stack(processed_v2)
        
        B, N, T = context_v1.shape
        
        # Combine batches: first B are view1, next B are view2
        # Shape becomes [2B, N, T]
        context = torch.cat([context_v1, context_v2], dim=0)
        
        # Flatten into Chronos-2 Variates form -> [2B*N, T]
        context = context.view(2 * B * N, T) 

        # 3. Create Group IDs
        # Each view instance must be distinctly grouped
        # view1 items get 0 to B-1; view2 items get B to 2B-1
        group_ids = torch.arange(2 * B).repeat_interleave(N)

        return {
            "context": context,        # [2B*N, T]
            "group_ids": group_ids,    # [2B*N]
            "labels": labels           # [B]
        }

# A Recipe Structure:
# (List of (file_path, start_time_idx), slice_idx, channel_shift, label_bits)
Recipe = Tuple[List[Tuple[str, int]], int, int, str]

class DataManifest:
    """
    Handles file discovery, splitting, and shape caching.
    """
    def __init__(self, root_dir: str, layouts: Optional[List[str]] = None, lpf_cutoff: Optional[float] = None):
        # Structure: layout_name -> { label_str -> [file_paths] }
        self.layout_buckets: Dict[str, Dict[str, List[str]]] = {}
        self.shape_cache: Dict[str, Tuple[int, ...]] = {} 
        self.input_layouts = layouts
        search_path = os.path.join(root_dir, "*", "*.npy")
        all_files = sorted(glob.glob(search_path))
        
        for fpath in all_files:
            layout_name = fpath.split(os.sep)[-2]
            
            # Filter by layout if provided
            if layouts is not None and layout_name not in layouts:
                continue

            # Filter by lpf_cutoff if provided
            if lpf_cutoff is not None and f"lpf{lpf_cutoff}" not in fpath:
                continue

            # Ensure layout bucket exists
            if layout_name not in self.layout_buckets:
                self.layout_buckets[layout_name] = {}

            # 2. Identify Label
            label_str = self._get_label_str(fpath) 
            
            # Ensure label bucket exists for this layout
            if label_str not in self.layout_buckets[layout_name]:
                self.layout_buckets[layout_name][label_str] = []
            self.layout_buckets[layout_name][label_str].append(fpath)

        # Cache num_chairs per layout (= length of any label string in that layout)
        self.num_chairs_per_layout: Dict[str, int] = {
            layout: len(next(iter(labels)))
            for layout, labels in self.layout_buckets.items()
            if labels
        }

    def get_file_shape(self, fpath: str) -> Tuple[int, ...]:
        if fpath in self.shape_cache: return self.shape_cache[fpath]
        try:
            shape = np.load(fpath, mmap_mode='r').shape
            self.shape_cache[fpath] = shape
            return shape
        except Exception:
            return (0, 0, 0, 0)

    def _get_label_str(self, path: str) -> str:
        """
        Parses the filename to identify which chairs are present and returns
        an N-bit string representation, where N is the number of chairs in the
        deployment (e.g. '101100' for a 6-chair deployment with chairs 1, 3, 4).

        Chair count detection (in order):
          1. Filename contains "XDesks" (case-insensitive) -> num_chairs = X.
          2. Fallback: num_chairs = 4.
        """
        path_lower = path.lower()

        # 1. Detect number of chairs from filename hints: e.g. '6Desks', '5Desks'
        desk_match = re.search(r'(\d+)desks', path_lower)
        if desk_match:
            num_chairs = int(desk_match.group(1))
        else:
            num_chairs = 4  # default fallback

        # 2. Find 'chair' followed by digits/underscores
        match = re.search(r'chair([\d_]+)', path_lower)
        if match:
            chair_suffix = match.group(1)
            # Build N-bit string: bit i is '1' if chair (i+1) appears in the suffix
            bits = ''.join('1' if str(i) in chair_suffix else '0'
                           for i in range(1, num_chairs + 1))
            return bits

        Warnings.warn(f"Could not determine label for path: {path}, defaulting to all zeros")
        return '0' * num_chairs

    def get_layouts(self) -> List[str]:
        if self.input_layouts is not None:
            return [layout for layout in self.input_layouts if layout in self.layout_buckets]
        return list(self.layout_buckets.keys())

    def get_files(self, layout: str, label: str) -> List[str]:
        return self.layout_buckets.get(layout, {}).get(label, [])


class SyntheticSignalDataset(Dataset):
    """
    Dataset that supports both combinatorial training and linear testing.
    """
    def __init__(self, 
                 manifest: DataManifest, 
                 n_channels: int = 4, 
                 window_size: int = 512,
                 stride: int = 256,
                 max_recipes: int = None,
                 stack_complex: bool = True,
                 synthesis_mode: bool = True,
                 aug_layout_training: bool = True,
                 aug_layout_testing: bool = False,
                 augment_phase: bool = False,
                 augment_phase_step: int = 60,
                 augment_time_warp: bool = False,
                 time_warp_num_knots: int = 6,
                 time_warp_strength: float = 15.0,
                 blending_alpha_enabled: bool = True,
                 blending_alpha_range: Tuple[float, float] = (0.1, 1.1),
                 min_max_normalization: bool = True,
                 convert_complex_to_float: str = "I_Q",
                 range_gating_width: int = 5,
                 augment_range_gating_offset: bool = False,
                 desk: Optional[List[List[int]]] = None,
                 supcon_mode: bool = False,
                 random_starting_index: bool = False,
                 mirroring_room: bool = False):
        """
        Args:
            manifest: Populated DataManifest.
            n_channels: Number of channels. When set to 1, no shifting is applied (no augmented layout).
            window_size: Length of time window to extract from each signal. (at 10Hz, 512 samples = ~51.2s)
            stride: Step size for sliding window extraction.
            max_recipes: Cap on dataset size (Only applies in synthesis_mode).
            stack_complex: Convert complex64 -> 2-channel float32.
            synthesis_mode: 
                If True (Train): Generates mixed combinations and random augmentations.
                If False (Test): Iterates strictly through existing files (no mixing, no shifting).
            aug_layout_testing:
                If True during testing, applies shifts to simulate layout augmentation.
            augment_phase:
                If True, applies random phase rotation augmentations during synthesis.
            augment_phase_step:
                Number of steps for phase augmentation (k * 2pi / step).
            augment_time_warp:
                If True, applies random time warping augmentations during synthesis.
            time_warp_num_knots:
                Number of knots for time warping spline.
            time_warp_strength:
                Strength of time warping (std dev of offsets).
            blending_alpha_enabled:
                If True, applies random alpha blending when mixing signals.
            blending_alpha_range:
                Range for random alpha blending factor (min, max).
            convert_complex_to_float:
                "I_Q" (default) or "mag_phase".
            range_gating_width:
                Width of the range gating window (number of bins to keep around the center).
            augment_range_gating_offset:
                If True, applies a random offset in [-1, 0, 1] to the center index per channel.
            desk:
                List of lists of integers specifying which desks to use for each layout. 
                If None, uses all. If a layout's list is empty ([]), uses all for that layout.
        """
        self.n_channels = n_channels
        self.window_size = window_size
        self.stride = stride
        self.stack_complex = stack_complex
        self.aug_layout_testing = aug_layout_testing
        self.aug_layout_training = aug_layout_training
        self.synthesis_mode = synthesis_mode
        self.augment_phase = augment_phase
        self.augment_phase_step = augment_phase_step
        self.augment_time_warp = augment_time_warp
        self.time_warp_num_knots = time_warp_num_knots
        self.time_warp_strength = time_warp_strength
        self.blending_alpha_enabled = blending_alpha_enabled
        self.blending_alpha_range = blending_alpha_range
        self.convert_complex_to_float = convert_complex_to_float
        self.recipes: List[Recipe] = []
        self.neg_pools = {}
        self.min_max_normalization = min_max_normalization
        self.range_gating_width = range_gating_width
        self.augment_range_gating_offset = augment_range_gating_offset
        self.manifest = manifest
        self.desk = desk
        self.supcon_mode = supcon_mode
        self.random_starting_index = random_starting_index
        self.mirroring_room = mirroring_room
        
        # --- Mode Switching ---
        if synthesis_mode:
            self._build_training_recipes(manifest, max_recipes)
        else:
            self._build_testing_recipes(manifest)
            
        print(f"Dataset initialized in mode={'TRAIN/SYNTH' if synthesis_mode else 'TEST/LINEAR'}")
        print(f"Total Samples: {len(self.recipes)}")

    def _get_file_segments(self, fpath: str) -> List[Tuple[str, int]]:
        """
        Given a file, returns ALL valid (file, start_idx) tuples.
        """
        shape = self.manifest.get_file_shape(fpath)
        if len(shape) < 3: return []
        
        t_dim = shape[2] # Assumes (4, 4, T, 3)
        if t_dim < self.window_size: return []
        
        if self.random_starting_index:
            return [(fpath, -1)]
        
        max_start = t_dim - self.window_size
        # Generate: [(f, 0), (f, 256), (f, 512)...]
        indices = range(0, max_start + 1, self.stride)
        return [(fpath, idx) for idx in indices]

    def _build_testing_recipes(self, manifest: DataManifest):
        """
        Linear scan: No mixing, no shifts. Just the raw data.
        Returns whatever is in the folder.
        """
        for layout_idx, layout in enumerate(manifest.get_layouts()):
            print("Layout is", layout, self.desk[layout_idx])
            # Get all labels/files in this layout
            labels = manifest.layout_buckets[layout].keys()
            for lbl in labels:
                files = manifest.get_files(layout, lbl)
                
                for fpath in files:
                    # Get all segments for this file
                    segments = self._get_file_segments(fpath)

                    # Each file contains num_chairs samples (slices 0..num_chairs-1)
                    # Label bits (e.g. '1000' or '100010') describe these slices.
                    num_chairs = manifest.num_chairs_per_layout.get(layout, 4)
                    for slice_idx in range(num_chairs):
                        if self.desk is not None and layout_idx < len(self.desk):
                            if len(self.desk[layout_idx]) > 0 and (slice_idx + 1) not in self.desk[layout_idx]:
                                continue
                        # Recipe:
                        # 1. File List: Just this one file, but with multiple segments
                        # 2. Slice: The current index
                        # 3. Shift: 0 (No augmentation for testing)
                        # 4. Label Bits: The file's native label
                        for seg in segments:
                            # seg is fpath, start_idx
                            if self.aug_layout_testing:
                                # Apply shifts to simulate layout augmentation
                                for shift in range(self.n_channels):
                                    for is_mirrored in ([False, True] if self.mirroring_room else [False]):
                                        self.recipes.append(([seg], slice_idx, shift, lbl, layout, False, is_mirrored))
                            else:
                                for is_mirrored in ([False, True] if self.mirroring_room else [False]):
                                    self.recipes.append(([seg], slice_idx, 0, lbl, layout, False, is_mirrored))
    
        self._print_statistics()

    def _build_training_recipes(self, manifest: DataManifest, max_recipes: int):
        """
        Single-file with Alpha Blended Augmentations (Dynamic Sampling).
        """
        import random
        for layout_idx, layout in enumerate(manifest.get_layouts()):
            num_chairs = manifest.num_chairs_per_layout.get(layout, 4)
            
            # Pre-compute pools of negative segments per slice_idx
            # For each slice_idx, we want all segments from files where label_bits[slice_idx] == '0'
            neg_pools = {i: [] for i in range(num_chairs)}
            
            all_segments_info = []

            for lbl in manifest.layout_buckets[layout].keys():
                files = manifest.get_files(layout, lbl)
                for fpath in files:
                    segments = self._get_file_segments(fpath)
                    for seg in segments:
                        all_segments_info.append((seg, lbl))
                        for i in range(num_chairs):
                            if lbl[i] == '0':
                                neg_pools[i].append(seg)
            
            self.neg_pools[layout] = neg_pools

            for seg, lbl in all_segments_info:
                for slice_idx in range(num_chairs):
                    if self.desk is not None and layout_idx < len(self.desk):
                        if len(self.desk[layout_idx]) > 0 and (slice_idx + 1) not in self.desk[layout_idx]:
                            continue
                    if self.aug_layout_training:
                        shift_range = range(self.n_channels)
                    else:
                        shift_range = [0]
                    for shift in shift_range:
                        for is_mirrored in ([False, True] if self.mirroring_room else [False]):
                            # Base recipe: single segment
                            self.recipes.append(([seg], slice_idx, shift, lbl, layout, False, is_mirrored))
                            
                            # Augmented recipe: flag it, keep it un-paired for now
                            # Skip adding explicit augmented recipes if supcon_mode is True,
                            # because SupCon generates matched augmented views pairs dynamically.
                            if not self.supcon_mode and len(neg_pools[slice_idx]) > 0:
                                self.recipes.append(([seg], slice_idx, shift, lbl, layout, True, is_mirrored))
        
        if max_recipes and len(self.recipes) > max_recipes:
            rng = np.random.default_rng(42)
            rng.shuffle(self.recipes)
            self.recipes = self.recipes[:max_recipes]
        
        self._print_statistics()
    
    def _print_statistics(self):
        """
        Iterates through metadata (recipes) to calculate class balance
        without loading heavy data.
        """
        label_0_count = 0
        label_1_count = 0
        
        # Iterate over metadata only (fast)
        for unpack in self.recipes:
            slice_idx = unpack[1]
            label_bits = unpack[3]
            # Check the bit string at the specific slice index
            if label_bits[slice_idx] == '1':
                label_1_count += 1
            else:
                label_0_count += 1
                
        total = label_0_count + label_1_count
        if total == 0:
            print("Warning: Dataset is empty.")
            return

        print("\n" + "="*30)
        print(f" Dataset Statistics ({'TRAIN' if hasattr(self, 'synthesis_mode') and self.synthesis_mode else 'TEST'})")
        print("="*30)
        print(f" Total Samples : {total}")
        print(f" Label 0 (Neg) : {label_0_count} ({label_0_count/total:.1%})")
        print(f" Label 1 (Pos) : {label_1_count} ({label_1_count/total:.1%})")
        
        # Optional: Calculate Positive Weight for Loss function
        # pos_weight = negative / positive
        if label_1_count > 0:
            suggested_pos_weight = label_0_count / label_1_count
            print(f" Suggested BCEWithLogitsLoss pos_weight: {suggested_pos_weight:.2f}")
        print("="*30 + "\n")

    def _has_overlap(self, bit_vals: List[int]) -> bool:
        acc = 0
        for v in bit_vals:
            if (acc & v) > 0: return True
            acc |= v
        return False

    def _safe_product(self, lists, limit=1000):
        # (Same randomized product logic as before)
        total_combos = 1
        for L in lists: total_combos *= len(L)
        if total_combos == 0: return []
        if total_combos <= limit: return list(itertools.product(*lists))
        
        rng = np.random.default_rng(42)
        combos = set()
        attempts = 0
        while len(combos) < limit and attempts < limit * 3:
            attempts += 1
            c = tuple(rng.choice(L) for L in lists)
            combos.add(c)
        return list(combos)

    def min_max_norm(self, signal: np.ndarray) -> np.ndarray:
        # signal: complex arrary, shape [C, T]
        r = np.abs(signal)
        r_min = np.min(r)
        r_max = np.max(r)
        eps = 1e-12
        r_norm = (r - r_min) / (r_max - r_min + eps)
        signal_norm = r_norm * np.exp(1j * np.angle(signal))
        return signal_norm

    def _apply_time_warp(self, signal: np.ndarray, strength: float = 15.0) -> np.ndarray:
        """
        Applies non-linear time warping to simulate speed variance.
        
        Args:
            signal: Shape [Channels, Time, Bins] or [Channels, Time]
            strength: Standard deviation of the random time shifts (in samples).
                      Higher = more variance.
        """
        # 1. Identify Time Axis and Dimensions
        # Signal is likely [C, T, B] coming from __getitem__ loop
        C, T, B = signal.shape
        
        # 2. Generate Random Flow Field
        # We define a few 'knots' (anchors) and perturb them
        num_knots = self.time_warp_num_knots
        # Original time points for knots (e.g., 0, 100, 200, ... 512)
        orig_knots = np.linspace(0, T-1, num_knots)
        
        # Random offsets: Start and End fixed at 0 to keep the window "anchored"
        # Middle points shift left/right to compress/stretch time locally
        offsets = np.random.normal(0, strength, num_knots)
        offsets[0] = 0 
        offsets[-1] = 0
        
        # Interpolate offsets to get a dense flow field for every time step 0..T-1
        x_grid = np.arange(T)
        dense_offsets = np.interp(x_grid, orig_knots, offsets)
        
        # Calculate sampling indices: index_new = index_old + flow
        sample_indices = x_grid + dense_offsets
        
        # Clamp to ensure we don't read outside valid time range
        sample_indices = np.clip(sample_indices, 0, T-1)
        
        # 3. Resample Signal
        # np.interp only works on 1D arrays, so we loop or flatten
        warped_signal = np.zeros_like(signal)
        
        # We apply the SAME warp to all Channels and Bins for physical consistency
        for c in range(C):
            for b in range(B):
                # Handle Complex Numbers (Real/Imag interpolated separately)
                if np.iscomplexobj(signal):
                    real = np.interp(sample_indices, x_grid, signal[c, :, b].real)
                    imag = np.interp(sample_indices, x_grid, signal[c, :, b].imag)
                    warped_signal[c, :, b] = real + 1j * imag
                else:
                    warped_signal[c, :, b] = np.interp(sample_indices, x_grid, signal[c, :, b])
                    
        return warped_signal

    def _finalize_signal(self, synthesized_signal: np.ndarray, shift: int, is_mirrored: bool) -> torch.Tensor:
        """
        Applies final transformations: circular shift, range gating, normalization, and complex to float conversion.
        """
        if is_mirrored and synthesized_signal.shape[0] > 3:
            synthesized_signal[[1, 3]] = synthesized_signal[[3, 1]]

        # Apply room mirroring
        if shift > 0:
            synthesized_signal = np.roll(synthesized_signal, shift, axis=0)

        center_idx = synthesized_signal.shape[-1] // 2
        half_width = self.range_gating_width // 2
        
        if self.augment_range_gating_offset:
            offsets = np.random.randint(-1, 2, size=synthesized_signal.shape[0])
        else:
            offsets = np.zeros(synthesized_signal.shape[0], dtype=int)
            
        gated_channels = []
        for c in range(synthesized_signal.shape[0]):
            c_idx = center_idx + offsets[c]
            start = max(0, c_idx - half_width)
            end = min(synthesized_signal.shape[-1], c_idx + half_width + 1)
            gated_channels.append(synthesized_signal[c, :, start:end])
            
        means = [np.mean(ch, axis=-1) for ch in gated_channels]
        synthesized_signal = np.stack(means, axis=0)

        if self.min_max_normalization:
            synthesized_signal = self.min_max_norm(synthesized_signal)

        if self.stack_complex:
            if self.convert_complex_to_float == "mag_phase":
                mag = torch.from_numpy(np.abs(synthesized_signal))
                angle = torch.from_numpy(np.angle(synthesized_signal))
                tensor_sig = torch.stack([mag, angle], dim=1).reshape(-1, *mag.shape[1:])
            else:
                real = torch.from_numpy(synthesized_signal.real)
                imag = torch.from_numpy(synthesized_signal.imag)
                tensor_sig = torch.stack([real, imag], dim=1).reshape(-1, *real.shape[1:])
        else:
            tensor_sig = torch.from_numpy(synthesized_signal)
            
        return tensor_sig, offsets

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        # 1. Unpack Recipe
        ingredients, slice_idx, shift, label_bits, layout, is_augmented, is_mirrored = self.recipes[idx]

        # Copy ingredients so we don't mutate the recipe over epochs
        ingredients = list(ingredients)
        
        # Label Logic
        target_label = 1.0 if label_bits[slice_idx] == '1' else 0.0
        label_tensor = torch.tensor(target_label, dtype=torch.float32)

        if self.supcon_mode:
            import random
            
            # Base logic handles a single positive segment inside `ingredients[0]`
            (fpath, start_idx) = ingredients[0]
            raw = np.load(fpath, mmap_mode='r')
            if start_idx == -1:
                start_idx = np.random.randint(0, max(1, raw.shape[2] - self.window_size - 100))
            end_idx = start_idx + self.window_size
            base_slice = raw[slice_idx, :, start_idx:end_idx, :].copy()
            
            # 1. Time Warp once for the base segment
            if self.augment_time_warp:
                base_slice = self._apply_time_warp(base_slice, strength=self.time_warp_strength)
            
            def augment_view(base_sig):
                view_sig = base_sig.copy()
                
                # Apply Random Phase
                if self.augment_phase: 
                    step = self.augment_phase_step
                    k = np.random.randint(0, step)
                    theta = k * (2 * np.pi / step)
                    phasor = np.exp(1j * theta)
                    view_sig = (view_sig * phasor).astype(np.complex64)
                    
                # Pair with randomly sampled negative segment unconditionally if available behind pool
                if layout is not None and len(self.neg_pools[layout][slice_idx]) > 0:
                    (neg_fpath, neg_start_idx) = random.choice(self.neg_pools[layout][slice_idx])
                    neg_raw = np.load(neg_fpath, mmap_mode='r')
                    if neg_start_idx == -1:
                        neg_start_idx = np.random.randint(0, max(1, neg_raw.shape[2] - self.window_size - 100))
                    neg_end = neg_start_idx + self.window_size
                    neg_slice = neg_raw[slice_idx, :, neg_start_idx:neg_end, :].copy()
                    
                    if self.augment_time_warp:
                        neg_slice = self._apply_time_warp(neg_slice, strength=self.time_warp_strength)
                        
                    if self.augment_phase:
                        k2 = np.random.randint(0, step)
                        theta2 = k2 * (2 * np.pi / step)
                        phasor2 = np.exp(1j * theta2)
                        neg_slice = (neg_slice * phasor2).astype(np.complex64)
                        
                    blending_alpha = 1.0
                    if self.blending_alpha_enabled:
                        low, high = self.blending_alpha_range
                        blending_alpha = np.random.uniform(low, high)
                    view_sig += blending_alpha * neg_slice
                
                return self._finalize_signal(view_sig, shift, is_mirrored)

            # Generate two augmentated views via branching logic
            view1, offsets1 = augment_view(base_slice)
            view2, offsets2 = augment_view(base_slice)
            
            meta_info = {
                "layout": layout if layout is not None else "unknown",
                "slice": slice_idx,
                "shift": shift,
                "offsets1": offsets1,
                "offsets2": offsets2,
                "is_mirrored": is_mirrored
            }
            return view1, view2, label_tensor, idx, meta_info

        # Default Flow (Classification)
        if is_augmented and layout is not None:
            import random
            pool = self.neg_pools[layout][slice_idx]
            if len(pool) > 0:
                ingredients.append(random.choice(pool))
        
        synthesized_signal = None
        
        # 1. Load & Sum
        for (fpath, start_idx) in ingredients:
            raw = np.load(fpath, mmap_mode='r')
            if start_idx == -1:
                start_idx = np.random.randint(0, max(1, raw.shape[2] - self.window_size - 100))
            end_idx = start_idx + self.window_size
            signal_slice = raw[slice_idx, :, start_idx:end_idx, :].copy()

            if self.augment_phase: 
                step = self.augment_phase_step
                k = np.random.randint(0, step)
                theta = k * (2 * np.pi / step)
                phasor = np.exp(1j * theta)
                signal_slice = (signal_slice * phasor).astype(np.complex64)
            
            if self.augment_time_warp:
                signal_slice = self._apply_time_warp(signal_slice, strength=self.time_warp_strength)
            
            if synthesized_signal is None:
                synthesized_signal = signal_slice
            else:
                blending_alpha = 1.0
                if self.blending_alpha_enabled:
                    low, high = self.blending_alpha_range
                    blending_alpha = np.random.uniform(low, high)
                synthesized_signal += blending_alpha * signal_slice
        
        tensor_sig, offsets = self._finalize_signal(synthesized_signal, shift, is_mirrored)
            
        meta_info = {
            "layout": layout if layout is not None else "unknown",
            "slice": slice_idx,
            "shift": shift,
            "offsets": offsets,
            "is_mirrored": is_mirrored
        }
        return tensor_sig, label_tensor, idx, meta_info
    
if __name__ == '__main__':
    print("--- Running Data Integrity Validation ---")
    manifest = DataManifest(root_dir="./tdma_sensing/cir_files/processed_cir", layouts=["deployment7"], lpf_cutoff=2.0)
    ds = SyntheticSignalDataset(manifest, n_channels=4, 
                                synthesis_mode=False, aug_layout_testing=False,
                                augment_phase=False, augment_time_warp=False,
                                window_size=1024, stride=256, max_recipes=20000, desk=[[]],
                                aug_layout_training=False, random_starting_index=True, mirroring_room=False)
    print(len(ds))
    for i in range(len(ds)):
        sig, lab, idx, meta = ds[i]
        ingredients, slice_idx, shift, label_bits, layout, is_augmented, is_mirrored = ds.recipes[i]
        print(f"Recipe {i+1}: Files: {ingredients}, Slice: {slice_idx}, Shift: {shift}, Label Bits: {label_bits}, Label: {lab.item()}, is augmented: {is_augmented}, is mirrored: {is_mirrored}, Meta: {meta}") 

    # manifest = DataManifest(root_dir='./tdma_sensing/cir_files/processed_cir', layouts=['deployment5'], lpf_cutoff=2.0)
    # ds = SyntheticSignalDataset(manifest, n_channels=4, synthesis_mode=True, 
    #                             supcon_mode=True, augment_phase=True, augment_time_warp=True)

    # print(f'Length: {len(ds)}')
    # view1, view2, lab, idx = ds[0]
    # print('Shape of view1:', view1.shape)
    # print('Shape of view2:', view2.shape)
    # print('Shape of lab:', lab.shape)