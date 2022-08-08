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
# from util.utils import make_bbox

def get_parser():
    parser = argparse.ArgumentParser(description="DACON LG")
    parser.add_argument("--config", type=str, default="config/LGAImers_test.yaml", help="config file")
    parser.add_argument("--weight", type=str, default=None, help="weight file")
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.weight is not None:
        cfg.weight = args.weight
    os.makedirs(cfg.save_path, exist_ok=True)
    copyfile(args.config, os.path.join(cfg.save_path, os.path.split(args.config)[-1]))
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
    logging.basicConfig(filename=os.path.join(output_log, "log.txt"), filemode="w")
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
    global logger, writer
    logger = get_logger()
    logger.info(args)
    writer = SummaryWriter(args.save_path)

    model = models.CustomModel(input_dim = 56, output_dim = 14)
    optimizer = create_optimizer(model.parameters(), args)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    lr_scheduler, args.epochs  = create_scheduler(args, optimizer)
    
    # CUDA
    model = torch.nn.DataParallel(model.cuda())

    # if args.weight:
    #     if os.path.isfile(args.weight):
    #         logger.info("=> loading weight '{}'".format(args.weight))
    #         checkpoint = torch.load(args.weight)
    #         model.load_state_dict(checkpoint["state_dict"], strict=False)

    #         # For loading a old version model
    #         if args.arch == "chestCAD_seg_cls" and len(
    #             checkpoint["state_dict"]["module.backbone._conv_stem_no_stride.weight"]
    #         ):
    #             model.module.backbone._conv_stem.weight = torch.nn.Parameter(
    #                 checkpoint["state_dict"]["module.backbone._conv_stem_no_stride.weight"]
    #             )
    #             model.module.backbone._conv_stem.bias = torch.nn.Parameter(
    #                 checkpoint["state_dict"]["module.backbone._conv_stem_no_stride.bias"]
    #             )

    #         logger.info("=> loaded weight '{}'".format(args.weight))
    #     else:
    #         logger.info("=> no weight found at '{}'".format(args.weight))


    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_data = dataset_LGAImers.Dataset_LGAImers(
        split="train",
        data_path=args.data_path_train,
        transform=None,
        features_X=56,
        features_Y=14,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    
    if args.evaluate:
        val_data = dataset_LGAImers.Dataset_LGAImers(
        split="val",
        data_path=args.data_path_val,
        transform=None,
        features_X=56,
        features_Y=14,
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    best_dict = {
        "loss": float("inf"),
        "best_idx": 1,
    }

    # loss_func = torch.nn.MSELoss()
    from loss.loss_lg_nrmse import lg_nrmse_loss
    loss_func = lg_nrmse_loss()
    loss_func.cuda()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        train_metrics = train(train_loader, model, optimizer, epoch, loss_func, lr_scheduler)

        for k, v in train_metrics.items():
            writer.add_scalar("train_" + k, 
                              v, epoch_log)

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + "/train_epoch_" + str(epoch_log) + ".pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename,
                _use_new_zipfile_serialization=False,
            )

            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + "/train_epoch_" + str(epoch_log - args.save_freq * 2) + ".pth"
                try:
                    os.remove(deletename)
                except:
                    print("no file at : " + deletename)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)
            # writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            # writer.add_scalar("lrx10", optimizer.param_groups[1]["lr"], epoch)

        if args.evaluate:
            val_metrics = validate(val_loader, model, loss_func)

            for k, v in val_metrics.items():
                writer.add_scalar("val_" + k, v, epoch_log)

            if best_dict[args.best_target] >= val_metrics[args.best_target]: # loss : 부등호 반대
                best_dict[args.best_target] = val_metrics[args.best_target]

                filename = os.path.join(args.save_path, "best{}.pth".format(best_dict["best_idx"]))
                torch.save(
                    {
                        "epoch": epoch_log,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    filename,
                    _use_new_zipfile_serialization=False,
                )
                best_dict["best_idx"] += 1
                best_dict["best_idx"] = (best_dict["best_idx"] % args.save_top_k) + 1
                logger.info("Saving best model: " + filename)



def train(train_loader, model, optimizer, epoch, loss_func, lr_scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.train()
    end_time = time.time()
    max_iter = args.epochs * len(train_loader)

    for batch_idx, (X_tensor, Y_tensor, df_ID) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        # X_tensor = torch.FloatTensor(X_list)
        # Y_tensor = torch.FloatTensor(Y_list)
        X_tensor = X_tensor.cuda(non_blocking=True)
        Y_tensor = Y_tensor.cuda(non_blocking=True)

        Y_pred = model(X_tensor)

        loss = loss_func(Y_pred, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), X_tensor.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        current_iter = epoch * len(train_loader) + batch_idx + 1
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=current_iter)

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % args.print_freq == 0:
            print_train_log(
                logger,
                epoch,
                args.epochs,
                batch_idx,
                len(train_loader),
                batch_time,
                data_time,
                remain_time,
                loss_meter,
            )
        
    logger.info("Train result at epoch [{}/{}]: loss {:.4f}.".format(epoch + 1, args.epochs, loss))
    train_metrics_dict = OrderedDict([("loss", loss_meter.avg)])

    return train_metrics_dict


def validate(val_loader, model, loss_func):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.eval()
    end_time = time.time()

    with torch.no_grad():
        for batch_idx, (X_tensor, Y_tensor, df_ID) in enumerate(val_loader):
            data_time.update(time.time() - end_time)
            # X_tensor = torch.FloatTensor(X_list)
            # Y_tensor = torch.FloatTensor(Y_list)
            X_tensor = X_tensor.cuda(non_blocking=True)
            Y_tensor = Y_tensor.cuda(non_blocking=True)

            Y_pred = model(X_tensor)

            loss = loss_func(Y_pred, Y_tensor)

            loss_meter.update(loss.item(), X_tensor.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            

            if (batch_idx + 1) % args.print_freq == 0:
                print_val_log(
                    logger,
                    batch_idx,
                    len(val_loader),
                    data_time,
                    batch_time,
                    loss_meter,
                )

        logger.info("Val result: loss {:.4f}.".format(loss))
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    val_metrics_dict = OrderedDict([("loss", loss_meter.val)])

    return val_metrics_dict


if __name__ == "__main__":
    main()
