# koalarization_torch
Deep Koalarization in PyTorch

![Grey-scale vs Ours vs Ground truth](https://github.com/Casper-Smet/koalarization_torch/blob/main/report/example.png)

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

## Training your own model

Follow the instructions in [./report/DATA.md](https://github.com/Casper-Smet/koalarization_torch/blob/main/report/DATA.md) to acquire your data, or substitute this with your own data (some commands might differ).

The default arguments are based on the file path described in [./report/DATA.md](https://github.com/Casper-Smet/koalarization_torch/blob/main/report/DATA.md). You should be able to use these without alterations if you followed the instructions. See the paragraph [Scripts](##Scripts) for the commands and their optional arguments for extra information.

### With default parameters
```bash
# Select a subset of images from the Kaggle data
# Places two files in ./data ("train_img.txt", "val_img.txt") containing file names (without file ext)
python -m koala.make_subset
# Train on the images and validation data from the previous step
# 5 epochs, 6 workers, batch size of 8, loads filenames from ./data/("train_img.txt", "val_img.txt")
python -m koala.train
# Simultaneously run Tensorboard to track training
tensorboard --logdir ./log/koala/
# Recolour all validation images using the latest trained model
# Uses validation data from ./data/val_img.txt, writes to ./data/
python -m koala.evaluate
```

### With experiment parameters
```bash
# Select a subset of images from the Kaggle data
python -m koala.make_subset --N 45000 --train_size 40000
# Train on the images and validation data from the previous step
python -m koala.train --epochs 40 --batch_size 50 --num_workers 8
# Simultaneously run Tensorboard to track training
tensorboard --logdir ./log/koala/
# Recolour all validation images using the latest trained model
python -m koala.evaluate -o ./data/output
```

![Results](https://github.com/Casper-Smet/koalarization_torch/blob/main/report/results_colourised.png)

## Scripts

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