"""
Adapted from CLAM: https://github.com/mahmoodlab/CLAM
"""
import argparse
from  dataset import Bag_Dataset,Instance_Dataset
import pytorch_lightning as pl
import os
import openslide
import torch
from torch.utils.data import DataLoader
from utils import *
import feature_dataset 
import math
from heatmap_utils import *
import csv


def infer_one_slide(args, model, device, features):
    """_summary_

    Args:
        args (argparse.Namespace): Arguments for the system
        model (torr.nn.Module): Model to infer
        device (torch.device): Device for infernce
        features (torch.Tensor): Features of the slide

    Returns:
        A (torch.Tensor): Heatmap
    """
    model.eval()
    features = features.to(device)
    with torch.no_grad():
        if args.model == "CLAM-SB" or args.model== "CLAM-MB":
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()
            if args.model == "CLAM-MB":
                A = A[Y_hat]
            A = A.view(-1, 1)
        elif args.model == "TransMIL":
            logits, Y_prob, Y_hat, return_dict = model(data = features.unsqueeze(0), return_attn=True)
            n = features.shape[0]
            add_length = (math.ceil(math.sqrt(n)))**2 -n
            n2 = n + add_length +1
            padding = 256 - (n2%256) if n2%256 > 0 else 0
            A = return_dict['A'][:,:,padding:(padding+n+1),padding:(padding+n+1)][:,:,0,:-1].view(8,-1,1)
        print(f"Predicted class: {Y_hat}")
    return A.cpu().numpy()
    
parser = argparse.ArgumentParser("Heatmap Inference script")
parser.add_argument("--model", type=str,choices=["CLAM-SB","CLAM-MB","TransMIL"],default="CLAM-SB")
parser.add_argument("--feature_ext",type=str,choices=["ResNet","KimiaNet", "DenseNet"],default="ResNet")
parser.add_argument("--ckpt_path",type=str, required=True)
parser.add_argument("--heatmap_dir",type=str,required=True)
parser.add_argument("--feat_dir",type=str,required=True)
parser.add_argument("--slide_dir",type=str,required=True)
parser.add_argument("--csv_path",type=str,default=None)
parser.add_argument("--gpu",type=bool,default=False)
parser.add_argument("--slide_ext",type=str,default=".ndpi")
parser.add_argument("--instance_loss",type=str,default="svm")
parser.add_argument("--n_classes",type=int,required=True)
parser.add_argument('--drop_out',action="store_true",default=False)
parser.add_argument("--vis_level",type=int, default=-1)
parser.add_argument("--alpha", type=float, default=0.4)
parser.add_argument("--blank_canvas", action="store_true", default=False)
parser.add_argument("--use_ref_scores", action="store_true", default=False)
parser.add_argument("--blur",action="store_true",default=False)
parser.add_argument("--binarize",action="store_true",default=False)
parser.add_argument("--binary_thresh",type=int, default=1)
parser.add_argument("--custom_downsample",type=int,default=1)
args = parser.parse_args()
if __name__=="__main__":
    device=torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = create_model(args,device)

    print(list(torch.load(args.ckpt_path,map_location=torch.device("cpu")).keys()))
    model.load_state_dict(torch.load(args.ckpt_path,map_location=torch.device("cpu")))

    model.to(device)
    os.makedirs(args.heatmap_dir,exist_ok=True)
    heatmaps_vis_args = heatmap_vis_args = {'convert_to_percentiles': not args.use_ref_scores, 'vis_level': args.vis_level, 'blur': args.blur, 'custom_downsample': args.custom_downsample}
    for dir in os.listdir(args.slide_dir):
        slide_id = dir.split('.')[0]
        os.makedirs(os.path.join(args.heatmap_dir,slide_id),exist_ok=True)
        path = os.path.join(args.feat_dir,slide_id)
        feature = torch.concat([torch.load(os.path.join(path,file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path)])
        coords = torch.concat([torch.tensor(torch.load(os.path.join(path,file), map_location=torch.device('cpu'))['coords']) for file in os.listdir(path)]).numpy()
        A = infer_one_slide(args,model,device,feature)
        if args.model != "TransMIL":
            heatmap_save_name = '{}_{}.jpg'.format(args.feature_ext,args.model)
            heatmap = drawHeatmap(A, coords, os.path.join(args.slide_dir,slide_id+args.slide_ext),  
                                    cmap="jet", alpha=args.alpha, **heatmap_vis_args, 
                                    binarize=args.binarize, 
                                    blank_canvas=args.blank_canvas,
                                    thresh=args.binary_thresh,
                                    overlap=0.75, 
                                    top_left=None, bot_right = None,seg_level=-1, sthresh=15, mthresh=11, close = 2, use_otsu=False, a_t=1,a_h=1,max_n_holes=20)
            heatmap.save(os.path.join(args.heatmap_dir, slide_id ,heatmap_save_name), quality=100)
        else:
            for h in range(A.shape[0]):
                heatmap_save_name = '{}_{}_head_{}.jpg'.format(args.feature_ext,args.model,h)
                heatmap = drawHeatmap(A[h], coords, os.path.join(args.slide_dir,slide_id+args.slide_ext),  
                                        cmap="jet", alpha=args.alpha, **heatmap_vis_args, 
                                        binarize=args.binarize, 
                                        blank_canvas=args.blank_canvas,
                                        thresh=args.binary_thresh,
                                        overlap=0.75, 
                                        top_left=None, bot_right = None,seg_level=-1, sthresh=15, mthresh=11, close = 2, use_otsu=False, a_t=1,a_h=1,max_n_holes=20)
                heatmap.save(os.path.join(args.heatmap_dir, slide_id ,heatmap_save_name), quality=100)
        