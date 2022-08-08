import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error 

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    
    # loss
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score


class lg_nrmse_loss(nn.Module):
    def __init__(self):
        super(lg_nrmse_loss, self).__init__()
        self.MSEloss = torch.nn.MSELoss()

    def forward(self, gt, preds):
        # 각 Y Feature별 NRMSE 총합
        # Y_01 ~ Y_08 까지 20% 가중치 부여
        all_nrmse = []
        # gt = gt.detach().cpu().numpy()
        # preds = pred.detach().cpu().numpy()
        
        # for idx in range(1,15): # ignore 'ID'
        for idx in range(0, 14):
            # rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
            
            rmse = self.MSEloss(gt[:,idx], preds[:,idx])
            # nrmse = rmse/np.mean(np.abs(gt[:,idx]))
            nrmse = rmse / torch.mean(torch.abs(gt[:,idx]))
            all_nrmse.append(nrmse)
        
        # loss
        # score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
        
        all_nrmse_tensor = torch.stack(all_nrmse, 0)
        score = 1.2 * torch.sum(all_nrmse_tensor[:8]) + 1.0 * torch.sum(all_nrmse_tensor[8:15])

        return score