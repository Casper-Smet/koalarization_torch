# koalarization_torch
Deep Koalarization in PyTorch


## Installing
In order to install Koala we supply a setup.cfg/pyproject.toml. The current Koalafications (dependencies) are based off of pytorch Python 3.9.

```bash
git clone git@github.com:Casper-Smet/koalarization_torch.git
cd koalarization_torch
pip install .
```

Koala very well could work on earlier/ater versions of Python/Pytorch. Install the dependencies yourself, and try this:
```bash
git clone git@github.com:Casper-Smet/koalarization_torch.git
cd koalarization_torch
pip install --no-dependencies . 
```

## Scripts

### Train
Trains the Colorization model on the dataset.
```bash
python -m koala.train -h
>> usage: train.py [-h] [--img_dir IMG_DIR] [--train_filenames TRAIN_FILENAMES] [--val_filenames VAL_FILENAMES]
                [--file_format FILE_FORMAT] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--drop_last DROP_LAST] [--device DEVICE]

Trains a PyTorch Koalarization network.

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Directory containing all images used.
  --train_filenames TRAIN_FILENAMES
                        Path to image names used in training.
  --val_filenames VAL_FILENAMES
                        Path to image names used in validation.
  --file_format FILE_FORMAT
                        Format for images (e.g, "{}.jpeg").
  --epochs EPOCHS       Amount of epochs.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --batch_size BATCH_SIZE
                        Amount of images per batch.
  --num_workers NUM_WORKERS
                        Number of workers used in data loading
  --drop_last DROP_LAST
                        Set dataset size to divisible number.
  --device DEVICE       Set device used for training (GPU)
```

### Make subset
Generates a subset of images based on the ImageNet dataset. See report/DATA.md for more.
```bash
python -m koala.make_subset -h
usage: make_subset.py [-h] [--path PATH] [--N N] [--train_size TRAIN_SIZE] [--output_path OUTPUT_PATH] [--seed SEED]

Extracts a set of image names from a ImageNet dataset.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to image labels
  --N N                 Total number of images
  --train_size TRAIN_SIZE
                        Total number of images used in training. Number of images used in validation is N - Train size
  --output_path OUTPUT_PATH
                        Path to where output is written
  --seed SEED           Random seed
```

### Evaluate
```bash
python -m koala.evaluate -h
>> usage: evaluate.py [-h] [--img_dir IMG_DIR] [--filenames FILENAMES] [--file_format FILE_FORMAT]
                   [--checkpoint CHECKPOINT] [--checkpoint_dir CHECKPOINT_DIR] [--output_path OUTPUT_PATH]
                   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR, -d IMG_DIR
                        Directory containing all images used.
  --filenames FILENAMES, -f FILENAMES
                        Path to image names.
  --file_format FILE_FORMAT
                        Format for images (e.g, "{}.jpeg)".
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        Path to model checkpoint
  --checkpoint_dir CHECKPOINT_DIR
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Directory where files are outputted
  --batch_size BATCH_SIZE
                        Amount of images per batch.
  --num_workers NUM_WORKERS
                        Number of workers used in data loading
  --device DEVICE       Set device used for training (GPU)
```