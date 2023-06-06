"""
Training and logging utilities 
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CustomOptim import create_optimizer
from models import TransMIL, clam
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from topk.svm import SmoothTop1SVM
from torchmetrics.functional import structural_similarity_index_measure, mean_squared_error, mean_absolute_error

import wandb


def create_model(args, device, feature_dim=1024):
    """_summary_

    Args:
        args (_type_): _description_
        device (_type_): _description_
        feature_dim (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    if args.model == "CLAM-SB" or args.model == "CLAM-MB":
        if args.instance_loss == "svm":
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            instance_loss_fn = instance_loss_fn.cuda(device)
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        if args.model == "CLAM-SB":
            model = clam.CLAM_SB(n_classes = args.n_classes, subtyping=True, instance_loss_fn=instance_loss_fn, dropout=args.drop_out,feature_dim=feature_dim)
        elif args.model == "CLAM-MB":
            model = clam.CLAM_MB(n_classes = args.n_classes, subtyping=True, instance_loss_fn=instance_loss_fn, dropout=args.drop_out,feature_dim=feature_dim)
    elif args.model == "TransMIL":
        model = TransMIL.TransMIL(args.n_classes, device,feature_dim=feature_dim)
    return model
    
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        """_summary_

        Args:
            n_classes (_type_): _description_
        """
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        """_summary_
        """
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        """_summary_

        Args:
            Y_hat (_type_): _description_
            Y (_type_): _description_
        """
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        """_summary_

        Args:
            Y_hat (_type_): _description_
            Y (_type_): _description_
        """
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        """_summary_

        Args:
            c (_type_): _description_

        Returns:
            _type_: _description_
        """
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def calculate_error(Y_hat, Y):
    """ Calculate Error

    Args:
        Y_hat (np.array): predicted class
        Y (np.array): actual class

    Returns:
        error (float): return error
    """
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


def make_weights_for_balanced_classes_split(dataset):
    """ Weights for Multinomial sampling

    Args:
        dataset (torch.utils.data.Dataset): dataset for which weights

    Returns:
        weight (torch.DoubleTensor): weights
    """
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  
    return torch.DoubleTensor(weight)

def train_loop_clam(epoch, model, loader, optimizer, n_classes=5, bag_weight=0.7, writer = None, loss_fn = nn.CrossEntropyLoss(), device = torch.device('cpu')):
    """_summary_

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        optimizer (_type_): _description_
        n_classes (int, optional): _description_. Defaults to 5.
        bag_weight (float, optional): _description_. Defaults to 0.7.
        writer (_type_, optional): _description_. Defaults to None.
        loss_fn (_type_, optional): _description_. Defaults to nn.CrossEntropyLoss().
        device (_type_, optional): _description_. Defaults to torch.device('cpu').
    """
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data.squeeze(0), label=label.squeeze(0), instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 100 == 0:
            # wandb.log({'batch': batch_idx, 'loss':loss_value,'instance_loss': instance_loss, 'weighted_loss': total_loss.item()})
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            print("writing class acc")
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        print("Writing loss error cluserting loss")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)




def validate_clam(epoch, model, loader, n_classes=5, writer = None, loss_fn = nn.CrossEntropyLoss(), device = torch.device('cpu'),early_stopping = None, results_dir = None):
    """_summary_

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        n_classes (int, optional): _description_. Defaults to 5.
        writer (_type_, optional): _description_. Defaults to None.
        loss_fn (_type_, optional): _description_. Defaults to nn.CrossEntropyLoss().
        device (_type_, optional): _description_. Defaults to torch.device('cpu').
        early_stopping (_type_, optional): _description_. Defaults to None.
        results_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data.squeeze(0), label=label.squeeze(0), instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "model.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False



def train_transmil(epoch , model, loader, device, optimizer, n_classes=5,loss_fn = None, writer = None):
    """_summary_

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        device (_type_): _description_
        optimizer (_type_): _description_
        n_classes (int, optional): _description_. Defaults to 5.
        loss_fn (_type_, optional): _description_. Defaults to None.
        writer (_type_, optional): _description_. Defaults to None.
    """
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _ = model(data= data)
        loss = loss_fn(logits,label)
        acc_logger.log(Y_hat,label)
        train_loss += loss.item()
        loss.backward()
        error = calculate_error(Y_hat, label)
        train_error += error
        optimizer.step()
        optimizer.zero_grad()
        if (batch_idx + 1) % 100 == 0:
                  print('batch {}, loss: {:.4f}, '.format(batch_idx, loss.item()) + 
                'label: {}, pred label: {}'.format(label.item(), Y_hat.item()))

    train_loss /= len(loader)
    train_error /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            print("writing class acc")
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        print("Writing loss error")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate_transmil(epoch, model, loader, n_classes=5, writer = None, loss_fn = nn.CrossEntropyLoss(), device = torch.device('cpu'),early_stopping = None, results_dir = None):
    """_summary_

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        n_classes (int, optional): _description_. Defaults to 5.
        writer (_type_, optional): _description_. Defaults to None.
        loss_fn (_type_, optional): _description_. Defaults to nn.CrossEntropyLoss().
        device (_type_, optional): _description_. Defaults to torch.device('cpu').
        early_stopping (_type_, optional): _description_. Defaults to None.
        results_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
  
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _ = model(data = data, label=label)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()
           
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "model.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes, device, model_type="CLAM-SB", conf_matrix_path = None, save_pred=None):
    """_summary_

    Args:
        model (_type_): _description_
        loader (_type_): _description_
        n_classes (_type_): _description_
        device (_type_): _description_
        model_type (str, optional): _description_. Defaults to "CLAM-SB".
        conf_matrix_path (_type_, optional): _description_. Defaults to None.
        save_pred (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_pred_labels = np.zeros(len(loader))


    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            if model_type == "CLAM-SB" or model_type=="CLAM-MB":
                logits, Y_prob, Y_hat, _, _ = model(data.squeeze(0))
            elif model_type == "TransMIL":
                logits, Y_prob, Y_hat, _ = model(data = data, label=label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_pred_labels[batch_idx] = Y_hat.item()
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    if conf_matrix_path:
        conf_matrix = confusion_matrix(all_labels,all_pred_labels)
        a = ConfusionMatrixDisplay(conf_matrix).plot()
        plt.savefig(fname = conf_matrix_path)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if save_pred:
        np.save(os.path.join(save_pred,"probs.npy"),all_probs)
        np.save(os.path.join(save_pred,"labels.npy"),all_labels)
        np.save(os.path.join(save_pred,"pred_labels.npy"),all_pred_labels)

    return  test_error, auc, acc_logger, aucs

def collate_features(batch):
    """_summary_

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    slide_ids = batch[0][2]
    return [img, coords, slide_ids]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """_summary_

    Args:
        output_path (_type_): _description_
        asset_dict (_type_): _description_
        attr_dict (_type_, optional): _description_. Defaults to None.
        mode (str, optional): _description_. Defaults to 'a'.

    Returns:
        _type_: _description_
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def seed_torch(seed, device):
    """_summary_

    Args:
        seed (_type_): _description_
        device (_type_): _description_
    """
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_numpy(seed):
    """_summary_

    Args:
        seed (_type_): _description_
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

