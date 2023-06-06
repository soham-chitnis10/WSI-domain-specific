"""
Fast script to extraction of features
"""
import pytorch_lightning as pl
from  dataset import Bag_Dataset,Instance_Dataset
from models.feature_model import Feature_extract
import argparse
import os
import openslide
from torch.utils.data import DataLoader, ConcatDataset
from utils import collate_features, save_hdf5
import writer 
import torch
import h5py

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.ndpi')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--model', type=str, choices = ["ResNet","KimiaNet", "DenseNet"],default="ResNet")
parser.add_argument('--devices', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers',type=int,default=20)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--strategy', type=str, default="ddp")
args = parser.parse_args()

if __name__ == "__main__":
    
    bags = Bag_Dataset(args.csv_path)
    all_dataloaders = []
    os.makedirs(args.feat_dir, exist_ok=True)
    for bag_idx in range(len(bags)):
        slide_id = bags[bag_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        wsi = openslide.open_slide(slide_file_path)
        patches_dataset = Instance_Dataset(wsi,h5_file_path,slide_id)
        kwargs = {'num_workers': args.num_workers} if torch.cuda.is_available() else {}
        patch_loader = DataLoader(dataset = patches_dataset, batch_size=args.batch_size, **kwargs,collate_fn=collate_features)
        all_dataloaders.append(patch_loader)

    pred_writer = writer.PredWriter(args.feat_dir)
    trainer = pl.Trainer(accelerator="gpu",devices = args.devices,callbacks=pred_writer,strategy=args.strategy)
    model = Feature_extract(args.model)
    prediction = trainer.predict(model,all_dataloaders)        
