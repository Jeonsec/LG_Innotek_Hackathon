import re
from .bce import BCELoss, BCELosswithLS
from .focal import FocalLoss
from .asymmetric import AsymmetricLoss
from .rocstar import ROCstarLoss
from .aucm import AUCM_MultiLabel
from .dice import DiceLoss


def create_loss(loss_fn):
    if loss_fn == "BCE":
        return BCELoss()
    elif loss_fn == "BCE_LS":
        return BCELosswithLS()        
    elif loss_fn == "FL" or loss_fn == "Focal":
        return FocalLoss()
    elif loss_fn == "ROCstar":
        return ROCstarLoss()
    elif loss_fn == "ASL" or loss_fn == "Asymmetric":
        return AsymmetricLoss()
    elif loss_fn == "AUCM":
        return AUCM_MultiLabel()
    elif loss_fn == "DICE":
        return DiceLoss()
    else:
        NotImplementedError
