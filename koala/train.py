import argparse
from datetime import datetime
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn, optim
from torch.cuda import get_device_properties
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from .data import (
    DEFAULT_IMG_DIR,
    DEFAULT_TRAIN,
    DEFAULT_VAL,
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


def norm(l_star: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Normalises the L*A*B* format using in Colorization to RGB.

    Args:
        l_star (torch.Tensor): Luminescence channel
        ab (torch.Tensor): A*B* channel

    Returns:
        torch.Tensor: RGB image
    """
    lab = torch.cat(((l_star + 1) * 50, ab * 127))
    rgb = lab_to_rgb(lab)
    return rgb


def plot(l_star: torch.Tensor, pab: torch.Tensor, tab: torch.Tensor) -> Figure:
    """Plots L*A*B* images to Figure for tensorboard.

    Args:
        l_star (torch.Tensor): Luminescence channel
        pab (torch.Tensor): Predicted A*B*
        tab (torch.Tensor): True A*B*

    Returns:
        Figure: Matplotlib figure with two images
    """
    pred = norm(l_star, pab).permute(1, 2, 0)
    true = norm(l_star, tab).permute(1, 2, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(pred)
    ax1.set_title("Predicted")
    ax2.imshow(true)
    ax2.set_title("True")

    return fig


def parse_args() -> argparse.ArgumentParser:
    """Arg parser for training.

    Returns:
        argparse.ArgumentParser: Arg parser
    """
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
        "--val_filenames",
        type=Path,
        default=DEFAULT_VAL,
        help="Path to image names used in validation.",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default=DEFAULT_FORMAT,
        help='Format for images (e.g, "{}.jpeg").',
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
    val_filenames: Path,
    file_format: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    device: str,
):
    """Main function for training network

    Args:
        img_dir (Path): Path to images
        train_filenames (Path): Path to text file with image names
        val_filenames (Path): Path to text file with image names
        file_format (str): File format
        epochs (int): Number of epochs in training
        learning_rate (float): Learning rate
        batch_size (int): Number of images per batch
        num_workers (int): Number of CPU workers for data collection
        drop_last (bool): Ensure len(filenames) / batch size is whole number
        device (str): CPU or GPU, device to be trained on
    """
    # Prefix for file paths
    now_str = str(datetime.now())
    # Tensorboard
    writer = SummaryWriter(rf"./log/koala/{now_str}")

    # Load datasets
    train_dataset = KoalaDataset(
        load_image_names(train_filenames)[:-1], img_dir, file_format
    )
    val_dataset = KoalaDataset(
        load_image_names(val_filenames)[:-1], img_dir, file_format
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=SHUFFLE,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=False,
    )

    # Initialise Neural network and send to GPU/CPU
    model = Colorization()
    model.to(device)

    # Initialise Loss and optimiser
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda a: a.requires_grad, model.parameters()), lr=learning_rate
    )

    running_loss = 0.0  # Keeps track of loss of Tensorboard
    total_batches = 0
    print(f"Training on:\t {get_device_properties(device)}.")

    for epoch in trange(epochs, unit="Epoch"):
        # Walk through data in batches
        for batch_num, (inputs, targets) in tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch}",
            unit="Batch",
            total=len(train_dataset) // batch_size,
            leave=False,
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Colourise images
            predicted = model(inputs)
            # Calculate loss and back propagate
            loss = criterion(predicted, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Write to tensorboard every 100 batches
            if (batch_num % 100) == 99:
                writer.add_scalar("train/loss", running_loss, total_batches + batch_num)
                running_loss = 0.0
                writer.add_figure(
                    "train/prediction vs. true",
                    plot(
                        inputs[0].detach().cpu(),
                        predicted[0].detach().cpu(),
                        targets[0].detach().cpu(),
                    ),
                    global_step=total_batches + batch_num,
                )

        total_batches += batch_num + 1
        # Validate
        if True:
            running_loss_val = 0.0
            for batch_num, (inputs, targets) in tqdm(
                enumerate(val_dataloader),
                desc=f"Val Epoch {epoch}",
                unit="Batch",
                total=len(val_dataset) // batch_size,
                leave=False,
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                predicted = model(inputs)
                loss = criterion(predicted, targets)

                running_loss_val += loss.item()

            writer.add_scalar("val/loss", running_loss_val, epoch)
            writer.add_figure(
                "val/prediction vs. true 1",
                plot(
                    inputs[1].detach().cpu(),
                    predicted[1].detach().cpu(),
                    targets[1].detach().cpu(),
                ),
                global_step=epoch,
            )
            writer.add_figure(
                "val/prediction vs. true 2",
                plot(
                    inputs[6].detach().cpu(),
                    predicted[6].detach().cpu(),
                    targets[6].detach().cpu(),
                ),
                global_step=epoch,
            )
            writer.add_figure(
                "val/prediction vs. true 3",
                plot(
                    inputs[7].detach().cpu(),
                    predicted[7].detach().cpu(),
                    targets[7].detach().cpu(),
                ),
                global_step=epoch,
            )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"./models/{now_str}-koala_model.pt",
    )
    writer.flush()


if __name__ == "__main__":
    parser = parse_args()
    main(**vars(parser.parse_args()))
