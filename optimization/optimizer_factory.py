from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from .sam import SAM


def create_optimizer(params, cfg):
    use_sam = 1 if cfg.optimizer_SAM else 0

    if cfg.optimizer == "SGD":
        if use_sam:
            return SAM(
                params,
                SGD,
                lr=cfg.base_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            return SGD(
                params,
                lr=cfg.base_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )

    elif cfg.optimizer == "Adadelta":
        if use_sam:
            return SAM(params, Adadelta, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        else:
            return Adadelta(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "Adagrad":
        if use_sam:
            return SAM(params, Adagrad, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        else:
            return Adagrad(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "Adam":
        if use_sam:
            return SAM(params, Adam, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        else:
            return Adam(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "RMSprop":
        if use_sam:
            return SAM(
                params,
                RMSprop,
                lr=cfg.base_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            return RMSprop(
                params,
                lr=cfg.base_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
    else:
        raise Exception("Unknown optimizer : {}".format(cfg.optimizer))
