"""Utils for Inception-Resnet-v2."""

from functools import cache
from itertools import repeat
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_DATA, load_image_names

DEFAULT_RESNET_LABELS = DEFAULT_DATA / Path(
    "./imagenet-object-localization-challenge/LOC_val_solution.csv"
)
DEFAULT_SYSNET_MAPPING = DEFAULT_DATA / Path(
    "./imagenet-object-localization-challenge/LOC_synset_mapping.txt"
)


def resnet_labels_to_tensor(
    id_label_pair: tuple[str, str], label_map: dict = None, num_classes: int = 1000
) -> tuple[any, torch.Tensor]:
    """Turns a id-labels pair into a id-tensor pair

    Args:
        id_label_pair (tuple[str, str]): id label pair
        label_map (dict, optional): map of labels to integers. Defaults to load_resnet_synset(DEFAULT_SYSNET_MAPPING).
        num_classes (int, optional): Number of classes. Defaults to 1000.

    Returns:
        tuple[any, Tensor]: id, tensor pair
    """
    _, labels = id_label_pair
    pairs = labels.split()[::5]
    if label_map is None:
        label_map = load_resnet_synset(DEFAULT_SYSNET_MAPPING)

    label_arr = np.zeros(num_classes)
    label_arr[[label_map[k] for k in pairs]] = 1
    return _, torch.tensor(label_arr, dtype=torch.float)


def load_resnet_labels(
    p: Path, skip_header: bool = True, label_map: dict = None
) -> dict[str, torch.Tensor]:
    """Loads imagenet labels and converts them to Inception-Resnet labels.

    Args:
        p (Path): Path to label file
        skip_header (bool, optional): Skip the .csv header. Defaults to True.

    Returns:
        dict[str, Tensor]: Mapping of label id's to tensors
    """
    lines = load_image_names(p)[:-1]
    if skip_header:
        lines = lines[1:]
    id_label_pairs = map(str.split, lines, repeat(","))
    if label_map is None:
        id_tensor_pairs = map(resnet_labels_to_tensor, id_label_pairs)
    else:
        id_tensor_pairs = map(
            resnet_labels_to_tensor, id_label_pairs, repeat(label_map)
        )

    return dict(id_tensor_pairs)


@cache
def load_resnet_synset(p: Path) -> dict[str, int]:
    """Loads the synset label mapping (id to integer)

    Args:
        p (Path): Path to synset file

    Returns:
        dict[str, int]: id, int
    """
    mappings = load_image_names(p)[:-1]
    return {line.split()[0]: i for i, line in enumerate(mappings)}


def main():
    labels = load_resnet_labels(DEFAULT_RESNET_LABELS)
    label = labels["ILSVRC2012_val_00003605"]
    print(label)


if __name__ == "__main__":
    main()
