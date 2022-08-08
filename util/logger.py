def print_train_log(
    logger,
    epoch,
    total_epoch,
    data_iter,
    data_len,
    batch_time,
    data_time,
    remain_time,
    loss_meter,
):
    logger.info(
        "Epoch: [{}/{}][{}/{}] "
        "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        "Remain {remain_time} "
        "Loss {loss_meter.val:.4f} ".format(
            epoch + 1,
            total_epoch,
            data_iter,
            data_len,
            batch_time=batch_time,
            data_time=data_time,
            remain_time=remain_time,
            loss_meter=loss_meter
        )
    )


def print_train_cls_log(
    logger,
    epoch,
    total_epoch,
    data_iter,
    data_len,
    batch_time,
    data_time,
    remain_time,
    loss_meter,
):
    logger.info(
        "Epoch: [{}/{}][{}/{}] "
        "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        "Remain {remain_time} "
        "Loss {loss_meter.val:.4f} ".format(
            epoch + 1,
            total_epoch,
            data_iter,
            data_len,
            batch_time=batch_time,
            data_time=data_time,
            remain_time=remain_time,
            loss_meter=loss_meter,
        )
    )


def print_val_log(logger, data_iter, dat_len, data_time, batch_time, loss_meter):
    logger.info(
        "Val: [{}/{}] "
        "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) ".format(
            data_iter + 1,
            dat_len,
            data_time=data_time,
            batch_time=batch_time,
            loss_meter=loss_meter,
        )
    )


def print_bbox_train_log(
    logger,
    epoch,
    total_epoch,
    data_iter,
    data_len,
    batch_time,
    data_time,
    remain_time,
    loss_meter_seg,
    loss_meter_miou,
    loss_meter_cls,
    loss_meter,
):
    logger.info(
        "Epoch: [{}/{}][{}/{}] "
        "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        "Remain {remain_time} "
        "Loss_focal_bbox {loss_meter_seg.val:.4f} "
        "Loss_miou {loss_meter_miou.val:.4f}"
        "LossCls {loss_meter_cls.val:.4f} "
        "Loss {loss_meter.val:.4f} ".format(
            epoch + 1,
            total_epoch,
            data_iter,
            data_len,
            batch_time=batch_time,
            data_time=data_time,
            remain_time=remain_time,
            loss_meter_seg=loss_meter_seg,
            loss_meter_miou=loss_meter_miou,
            loss_meter_cls=loss_meter_cls,
            loss_meter=loss_meter,
        )
    )


def print_bbox_val_log(logger, data_iter, dat_len, data_time, batch_time, loss_meter):
    logger.info(
        "Val: [{}/{}] "
        "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) ".format(
            data_iter + 1,
            dat_len,
            data_time=data_time,
            batch_time=batch_time,
            loss_meter=loss_meter,
        )
    )
