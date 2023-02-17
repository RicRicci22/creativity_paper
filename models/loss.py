import torch.nn.functional as F
import torch


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target, ignore_index=0)


def kl_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def tot_loss(output, target, mean=None, logvar=None):
    if mean is None or logvar is None:
        return cross_entropy_loss(output, target), None
    elif mean is not None and logvar is not None:
        return cross_entropy_loss(output, target), kl_loss(mean, logvar)
    else:
        raise Exception("Wrong arguments passed to loss function")
