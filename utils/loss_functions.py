import torch.nn as nn


def get_criterion(loss_type):
    if loss_type == 'L2':
        criterion = nn.MSELoss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion
