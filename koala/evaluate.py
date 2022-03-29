import argparse
from math import ceil
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from . import Colorization, KoalaDataset
from .data import DEFAULT_IMG_DIR, DEFAULT_VAL, load_image_names
from .train import BATCH_SIZE, DEFAULT_FORMAT, DEVICE, NUM_WORKERS, norm

DEFAULT_OUTPUT = Path("./data/output")


def _build_parser() -> argparse.ArgumentParser:
    """Commandline argument parser.

    Returns:
        argparse.ArgumentParser: Parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        "-d",
        type=Path,
        default=DEFAULT_IMG_DIR,
        help="Directory containing all images used.",
    )

    parser.add_argument(
        "--filenames",
        "-f",
        type=Path,
        default=DEFAULT_VAL,
        help="Path to image names.",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default=DEFAULT_FORMAT,
        help='Format for images (e.g, "{}.jpeg)".',
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default="latest",
        help="Path to model checkpoint",
    )

    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))

    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where files are outputted",
    )

    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Amount of images per batch."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers used in data loading",
    )

    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Set device used for training (GPU)"
    )
    return parser


def to_file(
    fp: Path, filename: Union[str, Path], l_star: torch.Tensor, ab: torch.Tensor
) -> None:
    """Converts image from CIELAB to sRGB, and writes to fp / filename

    Args:
        fp (Path): Path to image directory
        filename (Union[str, Path]): Filename
        l_star (torch.Tensor): Luminescence channel
        ab (torch.Tensor): A*B* channels
    """
    rgb_image = norm(l_star, ab)
    save_image(rgb_image, fp / filename)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if str(args.checkpoint) == "latest":
        model_path: Path = max(args.checkpoint_dir.iterdir())
    else:
        model_path: Path = args.checkpoint_dir / args.checkpoint

    args.output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path)

    model = Colorization()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    image_names = load_image_names(args.filenames)[:-1]

    img_dir = args.img_dir

    file_format = args.file_format
    ds = KoalaDataset(image_names, img_dir, file_format)
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_num, (inputs, targets) in tqdm(
            enumerate(dataloader),
            unit="Batch",
            total=ceil(len(ds) / args.batch_size),
            leave=True,
        ):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            # Colourise images
            predicted = model(inputs)
            # Write to file
            for i, (l_star, ab) in enumerate(
                zip(inputs, predicted), start=batch_num * args.batch_size
            ):
                to_file(args.output_path, file_format.format(i), l_star, ab)


if __name__ == "__main__":
    main()
