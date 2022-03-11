import argparse
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.cuda import get_device_properties
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from .data import (
    DEFAULT_IMG_DIR,
    DEFAULT_TRAIN,
    KoalaDataset,
    lab_to_rgb,
    load_image_names,
)
from .network import Colorization

DEFAULT_FORMAT = "{}.JPEG"
EPOCHS = 5
BATCH_SIZE = 8
NUM_WORKERS = 6
DROP_LAST = True
LEARNING_RATE = 0.001
SHUFFLE = True
PIN_MEMORY = True
PERSISTENT_WORKERS = True
DEVICE = "cuda"


def norm(l, ab):
    lab = torch.cat(((l + 1) * 50, ab * 127))
    rgb = lab_to_rgb(lab)
    return rgb


def plot(l, pab, tab):
    pred = norm(l, pab).permute(1, 2, 0)
    true = norm(l, tab).permute(1, 2, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(pred)
    ax1.set_title("Predicted")
    ax2.imshow(true)
    ax2.set_title("True")

    return fig


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trains a PyTorch Koalarization network."
    )

    parser.add_argument(
        "--img_dir",
        type=Path,
        default=DEFAULT_IMG_DIR,
        help="Directory containing all images used.",
    )
    parser.add_argument(
        "--train_filenames",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to image names used in training.",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default=DEFAULT_FORMAT,
        help='Format for images (e.g, "{}.jpeg".',
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Amount of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate."
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
        "--drop_last",
        type=bool,
        default=DROP_LAST,
        help="Set dataset size to divisible number.",
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Set device used for training (GPU)"
    )

    return parser


def main(
    img_dir: Path,
    train_filenames: Path,
    file_format: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    device: str,
):
    writer = SummaryWriter(r"./scratch/log/koala")

    dataset = KoalaDataset(load_image_names(train_filenames)[:-1], img_dir, file_format)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=SHUFFLE,
    )

    model = Colorization()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda a: a.requires_grad, model.parameters()), lr=learning_rate
    )

    running_loss = 0.0
    print(f"Training on:\t {get_device_properties(device)}.")
    total_batches = 0
    for epoch in trange(epochs, unit="Epoch"):
        for batch_num, (inputs, targets) in tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch}",
            unit="Batch",
            total=len(dataset) // BATCH_SIZE,
            leave=False,
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs)
            loss = criterion(predicted, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_num % 100) == 99:
                writer.add_scalar("train/loss", running_loss, total_batches + batch_num)
                running_loss = 0.0
                writer.add_figure(
                    "predictions vs. actuals",
                    plot(
                        inputs[0].detach().cpu(),
                        predicted[0].detach().cpu(),
                        targets[0].detach().cpu(),
                    ),
                    global_step=total_batches + batch_num,
                )

        total_batches += batch_num


if __name__ == "__main__":
    parser = parse_args()
    main(**vars(parser.parse_args()))
