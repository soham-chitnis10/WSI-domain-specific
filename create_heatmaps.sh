#!/bin/sh
#SBATCH --job-name=heatmap           # Job name
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --output=heatmap%j.out             # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=f20201723@goa.bits-pilani.ac.in
#SBATCH --cpu-freq=high
#SBATCH --cpus-per-task=10                  # Run a task on 10 cpus
#SBATCH --gres=gpu:1                        # Run a single GPU task
#SBATCH --mem=128GB                          # Use 32GB of memory.
#SBATCH --partition=normal               # Use dgx partition.
##SBATCH --account=f20201723

# Ckpt paths
ResNetCLAM_SB="/home/f20201723/project/results/final_reliability/ResNet/CLAM_SB/exp_1/model.pt"
ResNetCLAM_MB="/home/f20201723/project/results/final_reliability/ResNet/CLAM_MB/exp_16/model.pt"
ResNetTransMIL="/home/f20201723/project/results/final_reliability/ResNet/TransMIL/exp_19/model.pt"
KimiaNetCLAM_SB="/home/f20201723/project/results/final_reliability/KimiaNet/CLAM_SB/exp_13/model.pt"
KimiaNetCLAM_MB="/home/f20201723/project/results/final_reliability/KimiaNet/CLAM_MB/exp_7/model.pt"
KimiaNetTransMIL="/home/f20201723/project/results/final_reliability/KimiaNet/TransMIL/exp_12/model.pt"
DenseNetCLAM_SB="/home/f20201723/project/results/final_reliability/DenseNet/CLAM_SB/exp_19/model.pt"
DenseNetCLAM_MB="/home/f20201723/project/results/final_reliability/DenseNet/CLAM_MB/exp_0/model.pt"
DenseNetTransMIL="/home/f20201723/project/results/final_reliability/DenseNet/TransMIL/exp_16/model.pt"

# Config
path="/home/f20201723/project/demo"
data_path=$path/data/Oligodendroglioma_IDH-mutant
patch_path=$path/patch_data/Oligodendroglioma_IDH-mutant
feature_path=$path/feat_path/Oligodendroglioma_IDH-mutant
heatmap_dir=$path/heatmap_result/Oligodendroglioma_IDH-mutant
feature_ext="ResNet"
model="CLAM-MB"

cd CLAM-master
python create_patches_fp.py --source $data_path --save_dir $patch_path --patch_size 256 --preset glioma.csv --seg --patch --stitch --patch_level 1
echo "Patch extraction completed"
cd ..
python feature_extraction.py --data_h5_dir $patch_path --data_slide_dir $data_path --csv_path $patch_path/process_list_autogen.csv --model $feature_ext --feat_dir $feature_path/$feature_ext
echo "Feature extraction completed"
if [ "$feature_ext"=="ResNet" ]
then
    if [ "$model"=="CLAM-SB" ]
    then
        ckpt_path=$ResNetCLAM_SB
    elif [ "$model"=="CLAM-MB" ]
    then
        ckpt_path=$ResNetCLAM_MB
    elif [ "$model"=="TransMIL" ]
    then 
    ckpt_path=$ResNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
elif [ "$feature_ext"=="DenseNet" ]
then
    if [ "$model"=="CLAM-SB" ]
    then
        ckpt_path=$DenseNetCLAM_SB
    elif [ "$model"=="CLAM-MB" ]
    then
        ckpt_path=$DenseNetCLAM_MB
    elif [ "$model"=="TransMIL" ]
    then 
    ckpt_path=$DenseNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
elif [ "$feature_ext"=="KimiaNet" ]
then
    if [ "$model"=="CLAM-SB" ]
    then
        ckpt_path=$KimiaNetCLAM_SB
    elif [ "$model"=="CLAM-MB" ]
    then
        ckpt_path=$KimiaNetCLAM_SB
    elif [ "$model"=="TransMIL" ]
    then 
    ckpt_path=$KimiaNetTransMIL
    else
        echo "Error: Inavlid model"
    fi
else
    echo "Error: Invalid Feature extractor"
fi
python heatmaps.py --heatmap_dir $heatmap_dir --feat_dir $feature_path/$feature_ext/ --slide_dir $data_path --csv_path $patch_path/process_list_autogen.csv  --gpu=True --n_classes 5 --ckpt_path $ckpt_path  --model $model --feature_ext $feature_ext --drop_out