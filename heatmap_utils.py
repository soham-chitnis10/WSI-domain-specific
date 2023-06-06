"""
Utility for heatmaps:
Adapted from CLAM: https://github.com/mahmoodlab/CLAM
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from scipy.stats import percentileofscore

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    """_summary_

    Args:
        score (float): score
        ref (np.array): reference

    Returns:
        float: percentile
    """
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, seg_level=-1,sthresh=20,mthresh=7,close =0,a_t=100,a_h=8,max_n_holes=1,use_otsu=True,**kwargs):
    """_summary_

    Args:
        scores (_type_): _description_
        coords (torch.Tensor): Coordinates of patches
        slide_path (str, optional): WSI path. Defaults to None.
        wsi_object (WholeSlideImage, optional): WSI object. Defaults to None.
        vis_level (int, optional): _description_. Defaults to -1.
        seg_level (int, optional): _description_. Defaults to -1.
        sthresh (int, optional): _description_. Defaults to 20.
        mthresh (int, optional): _description_. Defaults to 7.
        close (int, optional): _description_. Defaults to 0.
        a_t (int, optional): _description_. Defaults to 100.
        a_h (int, optional): _description_. Defaults to 8.
        max_n_holes (int, optional): _description_. Defaults to 1.
        use_otsu (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    wsi_ref_downsample = wsi_object.level_downsamples[1]
    vis_patch_size = tuple((np.array((256,256)) * np.array(wsi_ref_downsample)).astype(int))
    kwargs["patch_size"] = vis_patch_size
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    if seg_level< 0:
        if len(wsi_object.level_dim) == 1:
                seg_level = 0
        else:
                wsi = wsi_object.getOpenSlide()
                seg_level = wsi.get_best_level_for_downsample(64)
    wsi_object.segmentTissue(seg_level=seg_level, sthresh=sthresh, sthresh_up = 255, mthresh=mthresh, close = close, use_otsu=use_otsu, 
                            filter_params={'a_t':a_t,'a_h':a_h,'max_n_holes':max_n_holes}, ref_patch_size=512, exclude_ids=[], keep_ids=[])
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    """_summary_

    Args:
        wsi_path (str): WSI path
        seg_mask_path (_type_, optional): _description_. Defaults to None.
        seg_params (_type_, optional): _description_. Defaults to None.
        filter_params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

