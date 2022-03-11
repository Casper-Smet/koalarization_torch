from functools import cache
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from torch import nn
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


class SquarePad(nn.Module):
    """Pads the input image into a homogenous shape.
    Adapted from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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


class Normalize(nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img * 2) - 1


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
            [SquarePad(), Resize(img_size), ConvertImageDtype(torch.float)]
        )
        self.grayscale = Grayscale(3)

    def __len__(self) -> int:
        return len(self.img_names)

    def get_img(self, idx: int) -> torch.Tensor:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def load_image_names(p: Path) -> list[str]:
    """Utility function to quickly load file names from path."""
    return p.read_text().split("\n")


def main():
    import matplotlib.pyplot as plt

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
