#!/bin/sh
# Ckpt paths
ResNetCLAM_SB="saved_models/resnet_clam_sb.pt"
ResNetCLAM_MB="saved_models/resnet_clam_mb.pt"
ResNetTransMIL="saved_models/resnet_transmil.pt"
KimiaNetCLAM_SB="saved_models/kimianet_clam_sb.pt"
KimiaNetCLAM_MB="saved_models/kimianet_clam_mb.pt"
KimiaNetTransMIL="saved_models/kimianet_transmil.pt"
DenseNetCLAM_SB="saved_models/densenet_clam_sb.pt"
DenseNetCLAM_MB="saved_models/densenet_clam_mb.pt"
DenseNetTransMIL="saved_models/densenet_transmil.pt"

# Config
# set the configs
path=ROOT_PATH
data_path=DATA_PATH
patch_path=PATCH_PATH
feature_path=FEATURE_PATH
heatmap_dir=HEATMAP_PATH

# ["ResNet","KimiaNet", "DenseNet"]
feature_ext="ResNet" 
# ["CLAM-SB", "CLAM-MB", "TransMIL"]
model="CLAM-MB"

cd CLAM
python create_patches_fp.py --source $data_path --save_dir $patch_path --patch_size 256 --preset glioma.csv --seg --patch --stitch --patch_level 1
echo "Patch extraction completed"
cd ..
python feature_extraction.py --data_h5_dir $patch_path --data_slide_dir $data_path --csv_path $patch_path/process_list_autogen.csv --model $feature_ext --feat_dir $feature_path/$feature_ext
echo "Feature extraction completed"
if [ $feature_ext = "ResNet" ];
then
    echo "selected ResNet"
    if [ "$model" = "CLAM-SB" ]
    then
        ckpt_path=$ResNetCLAM_SB
    elif [ "$model" = "CLAM-MB" ]
    then
        ckpt_path=$ResNetCLAM_MB
    elif [ "$model" = "TransMIL" ]
    then 
        ckpt_path=$ResNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
elif [ "$feature_ext" == "DenseNet" ]
then
    if [ "$model" == "CLAM-SB" ]
    then
        ckpt_path=$DenseNetCLAM_SB
    elif [ "$model"== "CLAM-MB" ]
    then
        ckpt_path=$DenseNetCLAM_MB
    elif [ "$model" == "TransMIL" ]
    then 
    ckpt_path=$DenseNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
elif [ "$feature_ext" == "KimiaNet" ]
then
    if [ "$model" == "CLAM-SB" ]
    then
        ckpt_path=$KimiaNetCLAM_SB
    elif [ "$model" == "CLAM-MB" ]
    then
        ckpt_path=$KimiaNetCLAM_SB
    elif [ "$model" == "TransMIL" ]
    then 
    ckpt_path=$KimiaNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
else
    echo "Error: Invalid Feature extractor"
fi
python heatmaps.py --heatmap_dir $heatmap_dir --feat_dir $feature_path/$feature_ext/ --slide_dir $data_path --csv_path $patch_path/process_list_autogen.csv  --gpu=True --n_classes 5 --ckpt_path $ckpt_path  --model $model --feature_ext $feature_ext --drop_out