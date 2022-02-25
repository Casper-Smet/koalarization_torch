from functools import cache
from itertools import repeat
from pathlib import Path

import numpy as np
from torch import float as tfloat
from torch import tensor
from torch.functional import Tensor
from torch.utils.data import Dataset

try:
    # There's a breaking change in the newest version of PILL that torchvision relies on
    from torchvision.transforms import Compose, ConvertImageDtype, Grayscale, Resize
except ImportError as ie:

    if "cannot import name 'PILLOW_VERSION'" in ie.msg:
        import PIL

        PIL.PILLOW_VERSION = PIL.__version__

        from torchvision.transforms import Compose, ConvertImageDtype, Grayscale, Resize
    else:
        raise ie

from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import pad

DEFAULT_DATA = Path("./data")
DEFAULT_TRAIN = DEFAULT_DATA / Path("val_img.txt")
DEFAULT_VAL = DEFAULT_DATA / Path("train_img.txt")
DEFAULT_IMG_DIR = DEFAULT_DATA / Path(
    "./imagenet-object-localization-challenge/imagenet_object_localization_patched2019.tar/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val/"
)
DEFAULT_RESNET_LABELS = DEFAULT_DATA / Path(
    "./imagenet-object-localization-challenge/LOC_val_solution.csv"
)
DEFAULT_SYSNET_MAPPING = DEFAULT_DATA / Path(
    "./imagenet-object-localization-challenge/LOC_synset_mapping.txt"
)


class SquarePad:
    """Pads the input image into a homogenous shape.
    Adapted from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
    """

    def __call__(self, image: Tensor) -> Tensor:
        """Pads the input image into a homogenous shape.

        Args:
            image (Tensor): Input image [channels, width, height]

        Returns:
            Tensor: Input image with padding where channels, width = height, width = height
        """
        _, w, h = image.size()
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 0, "constant")


class KoalaDataset(Dataset):
    """Default Dataset for this torch Koalarization.

    This dataset assumes you format the data in the way described in report/DATA.md. See github for more.
    """

    def __init__(
        self,
        img_names: list[str],
        img_dir: Path,
        ext_format: str,
        img_size: tuple[int, int] = (299, 299),
    ):
        """Default KoalaDataset.

        Args:
            img_names (list[str]): Image names on disk (e.g. ILSVRC2012_val_00009523)
            img_dir (Path): Path to image directory
            ext_format (str): image name formatter (e.g. "{}.JPEG")
            img_size (tuple[int, int], optional): Image dimensions. Defaults to (299, 299).
        """
        self.img_names = img_names
        self.img_dir = img_dir
        self.ext_format = ext_format
        self.transforms = Compose(
            [SquarePad(), Resize(img_size), ConvertImageDtype(tfloat)]
        )
        self.grayscale = Grayscale(3)

    def __len__(self) -> int:
        return len(self.img_names)

    def get_img(self, idx: int) -> Tensor:
        """Reads the image at index idx from file system into Tensor, transforms appropriately.

        Sub function for easier subclassing.

        Args:
            idx (int): image index

        Raises:
            FileNotFoundError: if file at path does not exist, raise before trying to read with torch.

        Returns:
            Tensor: original image (padded)
        """
        img_path = self.img_dir / self.ext_format.format(self.img_names[idx])
        if not img_path.exists():
            raise FileNotFoundError(f"Input image at {img_path} does not exist.")
        img = self.transforms(read_image(img_path.as_posix(), ImageReadMode.RGB))
        return img

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Reads the image at index idx from file system into Tensor, transforms appropriately.

        Args:
            idx (int): image index

        Raises:
            FileNotFoundError: if file at path does not exist, raise before trying to read with torch.

        Returns:
            tuple[Tensor, Tensor]: grayscaled image, original (padded)
        """
        img_y = self.get_img(idx)
        img_x = self.grayscale(img_y)

        return img_x, img_y


def resnet_labels_to_tensor(
    id_label_pair: tuple[str, str], label_map: dict = None, num_classes: int = 1000
) -> tuple[any, Tensor]:
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
    return _, tensor(label_arr, dtype=tfloat)


def load_resnet_labels(p: Path, skip_header: bool = True) -> dict[str, Tensor]:
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
    id_tensor_pairs = map(resnet_labels_to_tensor, id_label_pairs)
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


def load_image_names(p: Path) -> list[str]:
    """Utility function to quickly load file names from path."""
    return p.read_text().split("\n")


def main():
    labels = load_resnet_labels(DEFAULT_RESNET_LABELS)
    label = labels["ILSVRC2012_val_00003605"]
    print(label)
    train = load_image_names(DEFAULT_TRAIN)
    _ = load_image_names(DEFAULT_VAL)
    ds = KoalaDataset(train, DEFAULT_IMG_DIR, "{}.JPEG")
    img_x, img_y = ds[0]
    plt.imshow(img_y.permute(1, 2, 0))
    plt.show()
    plt.imshow(img_x.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
