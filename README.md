# Domain-Specific Pre-training Improves Confidence in Whole Slide Image Classification
![image](https://github.com/soham-chitnis10/WSI-domain-specific/assets/77532586/3fed2964-99a7-42be-baac-265cc125aaf7)

*Published at EMBC 2023.* [Preprint](https://arxiv.org/abs/2302.09833v2). [IEEE Explorer](https://ieeexplore.ieee.org/document/10340659)

**Abstract**: Whole Slide Images (WSIs) or histopathology images are used in digital pathology. WSIs pose great challenges to deep learning models for clinical diagnosis, owing to their size and lack of pixel-level annotations. With the recent advancements in computational pathology, newer multiple-instance learning-based models have been proposed. Multiple-instance learning for WSIs necessitates creating patches and uses the encoding of these patches for diagnosis. These models use generic pre-trained models (ResNet-50 pre-trained on ImageNet) for patch encoding. The recently proposed KimiaNet, a DenseNet121 model pre-trained on TCGA slides, is a domain-specific pre-trained model. This paper shows the effect of domain-specific pre-training on WSI classification. To investigate the effect of domain-specific pre-training, we considered the current state-of-the-art multiple-instance learning models, 1) CLAM, an attention-based model, and 2) TransMIL, a self-attention-based model, and evaluated the models' confidence and predictive performance in detecting primary brain tumors - gliomas. Domain-specific pre-training improves the confidence of the models and also achieves a new state-of-the-art performance of WSI-based glioma subtype classification, showing a high clinical applicability in assisting glioma diagnosis.


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
git clone https://github.com/mahmoodlab/CLAM.git
```

## Patch and Feature Extraction

Download the file from [here](https://drive.google.com/file/d/17e-83Ge9fWByoVBCYH4YXATIwCepEPMB/view?usp=sharing). Place the file in `CLAM/presets`

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

To train a model, use the following command. Since the experiment was conducted with multiple data and model seeds the option to set it is available.

```shell
python train_seeded.py --name WANDB_PROJECT_NAME --n_classes NUM_CLASSES --feat_dir FEATURE_DIR --csv CSV_PATH --feature_model FEATURE_MODEL --model MODEL --drop_out --early_stopping --opt OPTIMIZER --result_dir RESULT_DIR
```

## Evaluation

To evaluate a model, use the following command.

```shell
python --n_classes NUM_CLASSES --device GPU_DEVICE --feat_dir FEATURE_DIR --csv_path CSV_PATH --model_path MODEL_CHECKPOINT --model MODEL --result_dir RESULT_DIR
```

## Heatmaps

To generate heatmaps for a given set of slides for a specific model, use the following command.

Before running the script, please set the paths and desired configuration in the script itself. Model Checkpoints can be found in the `results` folder.

```shell
bash create_heatmaps.sh
```
