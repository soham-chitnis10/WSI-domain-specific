"""
Utililies for evaluation
"""
import argparse
import netcal.metrics as networkcal_metrics
import netcal.presentation as networkcal_pres
import torch
from utils import *
import numpy as np


def confidence_eval(args, model, dataloader, device, n_bins=20):
    """
    Measure the confidence of model's prediction
    """
    ground_truth = []
    confidence = []
    model.eval()
    for batch_idx, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            if args.model == "CLAM-SB" or args.model=="CLAM-MB":
                logits, Y_prob, Y_hat, _, _ = model(data.squeeze(0))
            elif args.model == "TransMIL":
                logits, Y_prob, Y_hat, _ = model(data = data, label=label)
            ground_truth.append(label.item())
            confidence.append(Y_prob.squeeze(0).cpu().numpy())
    ground_truth = np.array(ground_truth)
    confidence = np.array(confidence)
    # Measure Calibration of model
    ece = networkcal_metrics.ECE(n_bins)
    ece_measure = ece.measure(confidence, ground_truth)
    # Reliability Diagram
    diagram = networkcal_pres.ReliabilityDiagram(n_bins)
    return ground_truth, confidence, ece_measure, diagram

