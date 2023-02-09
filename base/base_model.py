import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable and non-trainable parameters
        """
        tot_parameters = sum([p.numel() for p in self.parameters()])
        trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return super().__str__() + '\nTotal parameters {} \nTrainable parameters: {}'.format(tot_parameters, trainable_params)
    
    @abstractmethod
    def train(self, dataloader, epochs):
        """
        Train pass logic
        :return: Model output
        """
        raise NotImplementedError
