import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel

class CreativityModel(BaseModel):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.ll = nn.Parameter(torch.zeros((self.hidden_size, self.vocab_size)))
        
    
    