#!/bin/sh
#SBATCH --job-name=feature_ext           # Job name
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --output=feature_ext%j.out             # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=f20201723@goa.bits-pilani.ac.in
#SBATCH --cpus-per-task=20                  # Run a task on 10 cpus
#SBATCH --gres=gpu:2                        # Run a single GPU task
#SBATCH --mem=128GB                          # Use 32GB of memory.
#SBATCH --partition=normal               # Use dgx partition.
##SBATCH --account=f20201723
date

path="/home/f20201723/glioma_subtyping"
data_path=$path/data
patch_path=$path/patch_data_20x
feature_model="convunext"
feature_path=$path/features_20x/convunext_new
output_path=$path/outputs/$feature_model/

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