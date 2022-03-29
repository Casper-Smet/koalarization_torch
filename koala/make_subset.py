import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extracts a set of image names from a ImageNet dataset."
    )

    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "./data/imagenet-object-localization-challenge/LOC_val_solution.csv"
        ),
        help="Path to image labels",
    )

    parser.add_argument("--N", type=int, default=10_000, help="Total number of images")
    parser.add_argument(
        "--train_size",
        type=int,
        default=9000,
        help="Total number of images used in training. Number of images used in validation is N - Train size",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("./data/"),
        help="Path to where output is written",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    labels_path = args.path

    df_labels = pd.read_csv(labels_path.absolute())
    df_labels.head()

    train_size = args.train_size

    if not args.output_path.is_dir():
        args.output_path.mkdir()

    # Select N images from dataset
    n = args.N
    if args.seed is None:
        df_subset = df_labels[["ImageId"]].sample(n)
    else:
        df_subset = df_labels[["ImageId"]].sample(n, random_state=42)

    df_train = df_subset[:train_size]
    df_train

    df_val = df_subset[train_size:]
    df_val

    # Sanity check for overlap
    assert set(df_val["ImageId"]).isdisjoint(df_train["ImageId"])

    val_path = args.output_path / Path("val_img.txt")
    df_val.to_csv(val_path, header=False, index=False)

    train_path = args.output_path / Path("train_img.txt")
    df_train.to_csv(train_path, header=False, index=False)


if __name__ == "__main__":
    main()
