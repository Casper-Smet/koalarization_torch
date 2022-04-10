# Time planning for recreating Koalarization in PyTorch

All throughout this period I will be writing my report.

## Week 1 (14/02 - 18/02):
 - [x] MUST - Make planning
 - [x] MUST - Install PyTorch locally
## Week 2 (21/02 - 25/02):
  - [x] SHOULD - Retrieve ImageNet dataset
  - [x] SHOULD - Select subset of 10K images to train on (reduced from original 60K)
  - [x] MUST - Setup up project structure
    - [x] MUST - Directories
    - [x] MUST - Setup.cfg
  - [x] MUST Inception-ResNet-v2
    - [x] SHOULD - Find and install/download Inception-ResNet-v2 in PyTorch (possibly/preferably pretrained)
## Week 3 (28/02 - 04/03):
  - [x] MUST - Eye-test Inception-ResNet-v2's performace
  - [x] MUST - Make encoder structure
  - [x] MUST - Fusion layer
  - [x] MUST - Decoder
  - [x] MUST - Train (will take some time):
    - [x] SHOULD - 90% of dataset for training
    - [x] COULD - Batch size of 20 images
    - [ ] COULD - If training takes too long / run out of VRAM, ask for access to HU compute cluster (maybe through Docker). Original paper uses NVidia Tesla K80, which has 24 GB of VRAM and 4992 CUDA cores vs my GTX 1070.
  - [ ] SHOULD - First revision of the paper for grading
## Week 4 (07/03 - 11/03):
  - [ ] MUST - Validation:
    - [x] SHOULD - 10% validation
    - [x] COULD - Replicate user acceptance Study
## Week 5 (14/03 - 18/03):
  - [ ] COULD - Ablation study
    - [ ] COULD - Figure out how to do an ablation study
    - [ ] COULD - Execute ablation study
  - [ ] SHOULD - Second revision of the paper for grading
  - [ ] COULD - Produce dockerfile for reproducibility
## Week 6 (21/03 - 27/03):
  - [x] MUST - Last revision of the paper
