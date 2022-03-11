import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.cuda import get_device_properties, is_available
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

BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = "cuda" if is_available() else "cpu"
EPOCHS = 5
NUM_WORKERS = 6
PIN_MEMORY = True
DROP_LAST = True
SHUFFLE = True
PERSISTENT_WORKERS = True


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


def main():
    writer = SummaryWriter(r"./scratch/log/koala")

    dataset = KoalaDataset(load_image_names(DEFAULT_TRAIN), DEFAULT_IMG_DIR, "{}.JPEG")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
        persistent_workers=PERSISTENT_WORKERS,
        shuffle=SHUFFLE,
    )

    model = Colorization()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda a: a.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    running_loss = 0.0
    print(f"Training on:\t {get_device_properties(DEVICE)}.")
    total_batches = 0
    for epoch in trange(EPOCHS, unit="Epoch"):
        for batch_num, (inputs, targets) in tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch}",
            unit="Batch",
            total=len(dataset) // BATCH_SIZE,
            leave=False,
        ):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
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
    main()
