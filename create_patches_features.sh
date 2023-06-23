#!/bin/sh
date
path="glioma_subtyping" # root
data_path=$path/data # path to data
patch_path=$path/patch_data_20x # path to save patches
feature_model="ResNet" # Feature extractor
feature_path=$path/features_20x/convunext_new # Path to save features
output_path=$path/outputs/$feature_model/ # Path to generated outputs

if [ ! -d $patch_path ]
then
    mkdir $patch_path
fi
if [ ! -d $feature_path ]
then
    mkdir -p $feature_path
fi
if [ ! -d $output_path ]
then
    mkdir -p $output_path
fi

for dir in $(ls $data_path)
do
    if [ ! -d $patch_path/$dir ]
    then
        cd CLAM
        python create_patches_fp.py --source $data_path/$dir --save_dir $patch_path/$dir --patch_size 256 --preset glioma.csv --seg --patch --stitch --patch_level 1 > $patch_path/$dir.txt
        echo "Patching completed:" $dir
        cd ..
    else
        echo "Patched data already exists"
    fi
    echo "Starting Feature extraction: "$dir

    if [ ! -d $feature_path/$dir ]
    then
        python feature_extraction.py --data_h5_dir $patch_path/$dir --data_slide_dir $data_path/$dir --csv_path $patch_path/$dir/process_list_autogen.csv --model $feature_model --feat_dir $feature_path/$dir > $output_path/$dir
        echo "Feature Extraction completed:" $dir
    else
        echo "Features already extracted"
    fi

done

date