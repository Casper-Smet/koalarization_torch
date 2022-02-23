from pathlib import Path

DEFAULT_DATA = Path("./data")
DEFAULT_TRAIN = DEFAULT_DATA / Path("val_img.txt")
DEFAULT_VAL = DEFAULT_DATA / Path("train_img.txt")


def load_image_names(p: Path) -> list[str]:
    return p.read_text().split("\n")


def main():
    train = load_image_names(DEFAULT_TRAIN)
    val = load_image_names(DEFAULT_VAL)
    input()
    pass


if __name__ == "__main__":
    main()
