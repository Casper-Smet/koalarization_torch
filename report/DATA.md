# Data

For the training of this model we use the ImageNet object localization challenge dataset from [Kaggle](https://kaggle.com/c/imagenet-object-localization-challenge).

We take the validation set of 45K images from this dataset, assuming that the resulting dataset is balanced. Furthermore, we split this last subset into a training and validation subset at a ratio of 8:1.

The file structure is unaltered from unpacking the compressed file from Kaggle:

![file structure for data](https://github.com/Casper-Smet/koalarization_torch/blob/main/report/file_structure.jpg)

If you place this folder into current working directory's ./data folder, most arguments need little to no alteration.

