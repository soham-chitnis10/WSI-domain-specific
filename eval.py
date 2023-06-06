"""
Code to evaluation
"""
import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from feature_dataset import Feature_bag_dataset
from models import TransMIL, clam
from sklearn.metrics import confusion_matrix
from topk.svm import SmoothTop1SVM
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import *


def main(args):
    path = args.feat_dir
    data_csv = args.csv_path
    device = torch.device(args.device)
    model = create_model(args, device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    test_dataset = Feature_bag_dataset(root=path, csv_path=data_csv, split='test')
    test_dataloader = DataLoader(test_dataset, num_workers=4)
    test_error, test_auc, acc_logger = summary(model,test_dataloader,n_classes=args.n_classes,device=device,model_type=args.model, conf_matrix_path=os.path.join(args.result_dir, 'conf_matrix_'+args.model+'.jpg'))
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument("--device",type=int, default=0)
parser.add_argument("--feat_dir", type=str, required=True)
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--bag_loss", type=str, default="cross-entropy")
parser.add_argument('--instance_loss', type=str, default="svm")
parser.add_argument('--model_path', type=str,default=None)
parser.add_argument('--model', type=str, choices=["CLAM", "TransMIL"], default="CLAM")
parser.add_argument('--result_dir', type=str, required=True)

args = parser.parse_args()

if __name__ == "__main__":
    main(args)