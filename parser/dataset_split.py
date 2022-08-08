import os
import argparse
import pandas as pd
import numpy as np
import sys
from itertools import chain
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

parser = argparse.ArgumentParser(description='TP data splitter)')
parser.add_argument('--label_csv_path',
                    default='./data/train.csv',
                    type=str,
                    help="Path of the csv file containing data label.")
parser.add_argument('-k', '--k_fold',
                    default= 2,
                    type=int,
                    help="K fold")
parser.add_argument('--output_directory',
                    default='./data/',
                    type=str,
                    help="Path where result file will be saved")


def main(args):
    print(os.path.abspath(args.label_csv_path))
    df= pd.read_csv(args.label_csv_path)
             
    # ---------------------------------Set groups -------------------------------------------
    groups = df['ID']
    gss = GroupShuffleSplit(n_splits=1, train_size=.9, random_state=42)

    # ---------------------------------Split data -------------------------------------------    
    for idx_1, (train_index, test_valid_index) in enumerate(gss.split(df, groups=groups)):
    #for idx_1, (trainValid_index, test_index) in enumerate(gss.split(X=df['Path'],y=df[label_header], groups=groups)):
    
        train_df = df.iloc[train_index]
        valid_df = df.iloc[test_valid_index]
        
        output_path = os.path.join(args.output_directory, "test_%s"%(str(idx_1)))
        os.makedirs(output_path, exist_ok=True)
        # test_df.drop('PatientID', axis=1).to_csv(os.path.join(output_path,"test.csv"), index=False)
        train_df.to_csv(os.path.join(output_path,"train.csv"), index=False)
        
        valid_df.to_csv(os.path.join(output_path, "valid.csv"), index=False)

  
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



