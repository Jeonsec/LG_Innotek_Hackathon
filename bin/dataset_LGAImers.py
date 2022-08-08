import os
import os.path
import random
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class Dataset_LGAImers(Dataset):
    def __init__(
        self,
        split="train",
        data_path='data/train.csv',
        transform=None,
        features_X=56,
        features_Y=14,
    ):
        self.split = split
        self.data_list = make_dataset_from_csv(split, data_path)

    def __len__(self):
        # if self.split == "train":
        #     sum = 0
        #     for i in self.data_list:
        #         sum += len(i)
        #     return sum
        # else:
        #     return len(self.data_list)
        # print(len(self.data_list))
        return len(self.data_list)


    def __getitem__(self, index):
        if self.split in ["train", "val"]:
            df = self.data_list
            df_ID = df['ID']
            df_X = df.filter(regex='X') # Input : X Featrue
            df_Y = df.filter(regex='Y') # Output : Y Feature
            
            del df
            
            X_list, Y_list = df_X.iloc[index].to_list(), df_Y.iloc[index].to_list()
            X_tensor = torch.FloatTensor(X_list)
            Y_tensor = torch.FloatTensor(Y_list)

            del X_list, Y_list

            return X_tensor, Y_tensor, df_ID

        elif self.split in ["test"]:
            df = self.data_list
            df_ID = df['ID']
            df_X = df.filter(regex='X') # Input : X Featrue
            
            del df
            
            X_list = df_X.iloc[index].to_list()
            df_ID = df_ID.iloc[index]
            
            X_tensor = torch.FloatTensor(X_list)

            del X_list

            return X_tensor, df_ID


def make_dataset_from_csv(split="train", data_path=None):
    assert split in ["train", "val", "test"]
    # print(os.path.abspath(data_path))
    if not os.path.isfile(data_path):
        raise (RuntimeError("Data file do not exist: " + data_path + "\n"))

    df = pd.read_csv(data_path)

    # print(df_X.head(), df_Y.head())
    # print(df.iloc[0].to_list())

    return df





    