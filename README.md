# WSI-domain-specific

## Installation

Create a conda environment and install the requirements

```shell
conda create --name domain-wsi --file requirements.txt
```

Now, install smooth-topk

```shell
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```

Install net:cal lib Python library

```shell
pip install netcal
```

For Patch extraction, clone CLAM into this repository

```shell
git https://github.com/mahmoodlab/CLAM.git
```

## Patch and Feature Extraction

Download file from [here](https://drive.google.com/file/d/1okHRlO5kvCFCp5YAB9F2jKBKO_MxpobA/view?usp=sharing). Place the file in `CLAM/presets`

The following directory structure is required for data


```bash
├── root
│   ├── data
│   │   ├── Class1
│   │   ├── Class2
│   │   └── ...
```
Before running the script, set the path in the script itself.

```shell
bash create_patches_features.sh
```

This creates a following directory structure

```bash
├── root
│   ├── data
│   │   ├── Class1
│   │   │   ├── Slide1.ndpi
│   │   │   ├── Slide2.ndpi
│   │   │   └── ...
│   │   ├── Class2
│   │   │   ├── Slide1.ndpi
│   │   │   ├── Slide2.ndpi
│   │   │   └── ...
│   │   └── ...
│   ├── patches
│   │   ├── masks
│   │   │   ├── Slide1.png
│   │   │   ├── Slide2.png
│   │   │   └── ...
│   │   ├── patches
│   │   │   ├── Slide1.h5
│   │   │   ├── Slide2.h5
│   │   │   └── ...
│   │   ├── stitches
│   │   │   ├── Slide1.png
│   │   │   ├── Slide2.png
│   │   │   └── ...
│   │   └── process_list_autogen.csv
│   ├── features
│   │   ├── Feature_model
│   │   │   ├── Class1
│   │   │   │   ├── Slide1
│   │   │   │   ├── Slide2
│   │   │   │   └── ...
│   │   │   ├── Class2
│   │   │   │   ├── Slide1
│   │   │   │   ├── Slide2
│   │   │   │   └── ...
│   │   │   └── ...
```

## Training

To train a model use the following command. Since the experiment was conducted with multiple data and model seeds the option to set it is available.

```shell
python train_seeded.py --name WANDB_PROJECT_NAME --n_classes NUM_CLASSES --feat_dir FEATURE_DIR --csv CSV_PATH --feature_model FEATURE_MODEL --model MODEL --drop_out --early_stopping --opt OPTIMIZER --result_dir RESULT_DIR
```

## Evaluation

To evaluate a model, use the following command

```shell
python --n_classes NUM_CLASSES --device GPU_DEVICE --feat_dir FEATURE_DIR --csv_path CSV_PATH --model_path MODEL_CHECKPOINT --model MODEL --result_dir RESULT_DIR
```

## Heatmaps

To generate heatmaps for a given set of slides for a specific model, use the following command

Before running script, please set the paths and desired configuration in the script itself.

```shell
bash create_heatmaps.sh
```
