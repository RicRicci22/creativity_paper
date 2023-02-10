import torch.nn.functional as F
import torch

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def kl_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

def tot_loss(output, target, mean, logvar):
    return cross_entropy_loss(output, target) + kl_loss(mean, logvar)