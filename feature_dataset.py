"""
Dataloader for Features slide level
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Feature_bag_dataset(Dataset):
    """
    Dataloader for Features at slide level
    """
    def __init__(self,root,csv_path, split = None, num_classes=5) -> None:
        """_summary_

        Args:
            root (str): root path
            csv_path (str): path to csv file
            split (str, optional): Split train, val or test. Defaults to None.
            num_classes (int, optional): number of classes. Defaults to 5.
        """
        super(Feature_bag_dataset,self).__init__()
        df = pd.read_csv(csv_path)
        self.root = root
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.split = split
        if self.split:
            self.df = self.split_dataset()
        self.num_classes = num_classes
        self.cls_slide_id_prep()
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        path_slide = os.path.join(self.root, self.df['subtype'][idx], self.df['name'][idx])
        features = torch.concat([torch.load(os.path.join(path_slide,file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path_slide)])
        return features, torch.tensor(self.df['label'][idx])


    def split_dataset(self):
        pat_ids = self.df.pat_id.unique()
        train_ids, rem_ids = train_test_split(pat_ids,test_size=0.3)
        val_ids, test_ids = train_test_split(rem_ids, test_size=0.5)
        train_df = self.df.loc[self.df['pat_id'].isin(train_ids)].reset_index(drop=True)
        val_df = self.df.loc[self.df['pat_id'].isin(val_ids)].reset_index(drop=True)
        test_df = self.df.loc[self.df['pat_id'].isin(test_ids)].reset_index(drop=True)
        if self.split == 'train':
            return train_df
        elif self.split == 'val':
            return val_df
        elif self.split == 'test':
            return test_df
        else:
            raise NotImplementedError

    def cls_slide_id_prep(self):
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.df['label'] == i)[0]
    
    def getlabel(self, ids):
        return self.df['label'][ids]

