import argparse
from cmath import e
import enum
import logging
from collections import OrderedDict
import os
import random
from shutil import copyfile
import sys
import time

import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from bin import dataset_LGAImers
from model import models
# from model.model_factory import create_model
# from loss.loss_factory import create_loss
from optimization.optimizer_factory import create_optimizer
from scheduler.scheduler_factory import create_scheduler

from util import config
from util.logger import print_train_log, print_val_log
from util.metrics import AverageMeter

import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config/LGAImers_inference.yaml", help="config file")
    parser.add_argument("--model_path", type=str, default='TRAINED_220803_FINETUNING_NRMSE_LOSS/best2.pth', help="model path")
    parser.add_argument("--save_path", type=str, default='data/test_result', help="test output path")
    parser.add_argument("--sample_submission_path", type=str, default='data/sample_submission.csv', help="sample_submission path")


    args = parser.parse_args()
    assert args.config is not None

    cfg = config.load_cfg_from_cfg_file(args.config)

    if args.model_path is not None:
        cfg.model_path = args.model_path

    if args.save_path is not None:
        cfg.save_path = args.save_path

    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    output_log = os.path.join(args.save_path, "log")
    os.makedirs(output_log, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_log, "testlog.txt"), filemode="w")
    return logger


def set_seed(manual_seed):
    if manual_seed is not None:
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def main():
    global args
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)

    set_seed(args.manual_seed)
    main_worker()


def main_worker():
    global logger
    logger = get_logger()
    logger.info(args)

    model = models.CustomModel(input_dim = 56, output_dim = 14)
    # CUDA
    model = torch.nn.DataParallel(model.cuda())

    model_path = args.weight
    if "model_path" in args:
        model_path = args.model_path

    if model_path:
        if os.path.isfile(model_path):
            logger.info("=> loading weight '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["state_dict"], strict=False)

            logger.info("=> loaded weight '{}'".format(model_path))
        else:
            logger.info("=> no weight found at '{}'".format(model_path))

        test_data = dataset_LGAImers.Dataset_LGAImers(
            split="test",
            data_path=args.data_path_test,
            transform=None,
            features_X=56,
            features_Y=14,
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        test(test_loader, model)


def test(test_loader, model):
    logger.info(">>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>")
    out_dir = args.save_path
    out_csv_path = os.path.join(out_dir, "test_result.csv")

    model.eval()
    sample_submission_csv = pd.read_csv('data/sample_submission.csv')

    col_name_list = ['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07', 'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14']

    result_csv_f = os.path.join(out_dir, "test_result.csv")

    row_index = 0

    with torch.no_grad():
        for batch_idx, (X_tensor, df_ID) in enumerate(test_loader):

            X_tensor = X_tensor.cuda(non_blocking=True)
            Y_pred = model(X_tensor)
            Y_pred_np = Y_pred.cpu().detach().numpy()

            # print(df_ID)
            # print(Y_pred_np)
            
            for idx, ID_num in enumerate(df_ID):
                row = sample_submission_csv.loc[row_index]
                if row['ID'] == ID_num:
                    for idx_col, col_name in enumerate(col_name_list):
                        row[col_name] = Y_pred_np[idx, idx_col]

                    sample_submission_csv.loc[row_index] = row

                row_index += 1

                if (row_index % 100) == 0:
                    print(row_index)
            # ######
            # for idx_row, row in sample_submission_csv.iterrows():
            #     flag_done = False
            #     for j, ID_str in enumerate(df_ID):
            #         if row['ID'] == ID_str:
            #             for idx_col, col_name in enumerate(col_name_list):
            #                 row[col_name] = Y_pred_np[j, idx_col]
            #             sample_submission_csv.loc[idx_row] = row
            #             flag_done = True
            #     if flag_done:
            #         if idx_row % 100 == 0:
            #             print(idx_row)
            #             # print(row)
            #         continue
            # #######




        

    print(sample_submission_csv.head())
    sample_submission_csv.to_csv(result_csv_f, index=False)

    return


if __name__ == "__main__":
    main()