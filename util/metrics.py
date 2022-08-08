import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_iou(output, target):
    with torch.no_grad():
        one = torch.ones(output.shape, dtype=torch.uint8, device="cuda")
        zero = torch.zeros(output.shape, dtype=torch.uint8, device="cuda")

        binary_output = torch.tensor(output.shape, dtype=torch.uint8, device="cuda")
        binary_target = torch.tensor(output.shape, dtype=torch.uint8, device="cuda")

        binary_output = torch.where(output > 0.5, one, zero).transpose_(0, 1)
        binary_target = torch.where(target > 0.5, one, zero).transpose_(0, 1)

        binary_intersection = (binary_output & binary_target).sum(
            [1, 2, 3], keepdim=True
        )
        binary_intersection.squeeze_()

        binary_Union = (binary_output | binary_target).sum([1, 2, 3], keepdim=True)
        binary_Union.squeeze_()

        binary_target = binary_target.sum([1, 2, 3], keepdim=True)
        binary_target.squeeze_()

        binary_intersection = binary_intersection.detach().cpu().numpy()
        binary_Union = binary_Union.detach().cpu().numpy()
        binary_target = binary_target.detach().cpu().numpy()

    return binary_intersection, binary_Union, binary_target

def get_selective_iou(output, target):
    with torch.no_grad():
        one = torch.ones(output.shape, dtype=torch.uint8, device="cuda")
        zero = torch.zeros(output.shape, dtype=torch.uint8, device="cuda")

        binary_output = torch.tensor(output.shape, dtype=torch.uint8, device="cuda")
        binary_target = torch.tensor(output.shape, dtype=torch.uint8, device="cuda")

        #binary_output = torch.where(output > 0.5, one, zero).transpose_(0, 1)
        #binary_target = torch.where(target > 0.5, one, zero).transpose_(0, 1)
        
        binary_output = torch.where(output > 0.5, one, zero)
        binary_target = torch.where(target > 0.5, one, zero)

        binary_intersection = (binary_output & binary_target).sum([1, 2], keepdim=True)        
        binary_intersection.squeeze_()

        binary_Union = (binary_output | binary_target).sum([1, 2], keepdim=True)
        binary_Union.squeeze_()

        binary_target = binary_target.sum([1, 2], keepdim=True)
        binary_target.squeeze_()

        binary_intersection = binary_intersection.detach().cpu().numpy()
        binary_Union = binary_Union.detach().cpu().numpy()
        binary_target = binary_target.detach().cpu().numpy()

    return binary_intersection, binary_Union, binary_target
