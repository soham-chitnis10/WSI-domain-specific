"""
Code for training
"""
import argparse
import csv
import os
import re
import time

import feature_dataset as dataset
import numpy as np
import pandas as pd
import torch
from CustomOptim import *
from models import TransMIL, clam
from topk.svm import SmoothTop1SVM
from torch.utils.data import (DataLoader, SequentialSampler,
                              WeightedRandomSampler)
from torch.utils.tensorboard import SummaryWriter
from utils import *
from eval_utils import *

import wandb


def main(args):
    wandb.login()
    with wandb.init(project= args.name, config=vars(args), sync_tensorboard=True):
        stime = time.time()
        path = args.feat_dir
        data_csv = args.csv_path
        device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available()  else torch.device("cpu")
        writer = SummaryWriter(args.log_dir)
        loss_fn = torch.nn.CrossEntropyLoss()

        for data_seed in range(args.data_seeds):
            seed_numpy(data_seed)
            train_dataset = dataset.Feature_bag_dataset(root=path, csv_path = data_csv, split = 'train')
            weights = make_weights_for_balanced_classes_split(train_dataset)
            train_dataloader = DataLoader(train_dataset, num_workers=4, sampler = WeightedRandomSampler(weights,len(weights)))
            val_dataset = dataset.Feature_bag_dataset(root=path,csv_path = data_csv, split='val')
            val_dataloader = DataLoader(val_dataset, num_workers=4, sampler = SequentialSampler(val_dataset))
            test_dataset = dataset.Feature_bag_dataset(root=path, csv_path=data_csv, split='test')
            test_dataloader = DataLoader(test_dataset, num_workers=4)
            for model_seed in range(args.model_seeds):
                print(f"Exp:{data_seed}_{model_seed}")
                seed_torch(model_seed,device)
                model = create_model(args, device,test_dataset[0][0].shape[1])
                model = model.to(device)
                print(model)
                val_error, val_auc, _, _= summary(model, test_dataloader, args.n_classes, device, model_type = args.model)
                print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
                wandb.watch(model, log_freq=100)
                optimizer = create_optimizer(args, model,args.model=="TransMIL")
                result_dir = os.path.join(args.result_dir,'exp_'+str(data_seed)+'_'+str(model_seed))
                os.makedirs(result_dir, exist_ok=True)
                exp_idx = (model_seed+1) + (data_seed*3)
                if args.early_stopping:
                    early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)
                else:
                    early_stopping = None
                for epoch in range(args.epochs):
                    if args.model == "CLAM-SB" or args.model == "CLAM-MB":
                        print(f"Starting Training {epoch}")
                        train_loop_clam(epoch,model,train_dataloader,optimizer,n_classes=args.n_classes,bag_weight=args.bag_weight,writer=writer,device=device)
                        print(f"Starting Validation {epoch}")
                        stop = validate_clam(epoch,model,val_dataloader,n_classes=args.n_classes,writer=writer,device=device,early_stopping=early_stopping, results_dir =result_dir)
                    elif args.model == "TransMIL":    
                        print(f"Starting Training {epoch}")
                        train_transmil(epoch,model,train_dataloader,device, optimizer=optimizer,n_classes=args.n_classes, loss_fn=loss_fn, writer=writer)
                        print(f"Starting Validation {epoch}")
                        stop = validate_transmil(epoch, model, val_dataloader, n_classes=args.n_classes, device = device,writer=writer,early_stopping=early_stopping, results_dir=result_dir)
                    else:
                        raise NotImplementedError
                    if stop:
                        break
                if args.early_stopping:
                    eval_model = create_model(args, device,test_dataset[0][0].shape[1])
                    eval_model.to(device)
                    eval_model.load_state_dict(torch.load(os.path.join(result_dir, "model.pt")))
                else:
                    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
                    eval_model = model
                    eval_model.to(device)

                val_error, val_auc, _, _= summary(eval_model, val_dataloader, args.n_classes, device, model_type = args.model)
                print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

                test_error, test_auc, acc_logger, aucs = summary(eval_model, test_dataloader, args.n_classes, device, model_type=args.model, conf_matrix_path=os.path.join(result_dir, 'conf_matrix_'+args.model+'.jpg'), save_pred=result_dir)
                print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

                ground_truth, confidence, ece_meaure, diagram = confidence_eval(args, eval_model, test_dataloader, device, n_bins = args.bins)
                print(f'ECE measure:{ece_meaure}')
                diagram.plot(confidence, ground_truth, filename=os.path.join(result_dir,"diagram.jpg"))
                
                for i in range(args.n_classes):
                    print('class {}: auc: {}'.format(i,aucs[i]))

                    if writer and aucs[i] is not None:
                        writer.add_scalar('final/test_class_{}_auc'.format(i), aucs[i], exp_idx)

                for i in range(args.n_classes):
                    acc, correct, count = acc_logger.get_summary(i)
                    print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

                    if writer and acc is not None:
                        writer.add_scalar('final/test_class_{}_acc'.format(i), acc, exp_idx)

                if writer:
                    writer.add_scalar('final/val_error', val_error, exp_idx)
                    writer.add_scalar('final/val_overall_auc', val_auc, exp_idx)
                    writer.add_scalar('final/test_error', test_error, exp_idx)
                    writer.add_scalar('final/test_overall_auc', test_auc, exp_idx)
                    writer.close()
                
        end_time = time.time()
        print(f"Time taken: {end_time-stime}")

parser = argparse.ArgumentParser("Training model")
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument("--feat_dir", type=str, required=True)
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--feature_model", type=str, choices=["ResNet","KimiaNet","DenseNet","efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3","efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7",'efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','convnext_tiny','convnext_small','convnext_base','convnext_large', "convunext"],default="ResNet")
parser.add_argument("--model", type=str, choices=["CLAM-SB","CLAM-MB","TransMIL"],default="CLAM-SB")
parser.add_argument("--bag_loss", type=str, default="cross-entropy")
parser.add_argument('--instance_loss', type=str, default="svm")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument("--bag_weight", type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--opt', type=str, default="lookahead_radam")
parser.add_argument("--early_stopping", action='store_true', default=False)
parser.add_argument("--result_dir", type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--drop_out',action="store_true",default=False)
parser.add_argument("--bins", type=int, default=20)
parser.add_argument("--data_seeds", type=int, default=5)
parser.add_argument("--model_seeds", type=int, default=3)

args = parser.parse_args()

if __name__ == "__main__":
    main(args)