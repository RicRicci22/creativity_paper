import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from torchvision.models import resnet18, ResNet18_Weights

class CreativityModel(BaseModel):
    def __init__(self, backbone_name, hidden_size, latent_size, vocab_size):
        super().__init__()
        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.encoder = CreativityEncoder(backbone_name=self.backbone_name, hidden_size=self.hidden_size, vocab_size=self.vocab_size)
        self.vae = VaE(hidden_size=hidden_size, latent_size=latent_size)

        self.ll = nn.Parameter(torch.zeros((self.hidden_size, self.vocab_size)))
    
    def forward(self, images, questions, mode="train"):
        # Embed the questions
        embedded_questions = self.embedding(questions)
        # Encode the images and questions
        hiddens = self.encoder(images, embedded_questions)
        # Project to VaE latent space
        hiddens = self.vae(hiddens, mode=mode)

        #logits = torch.matmul(encoded_images, self.ll)
        #return logits
        return hiddens
        
    

class CreativityEncoder(nn.Module):

    '''
    Encoder for creativity model.
    It takes in a batch of image features, and a batch of tokenized questions. 
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    '''

    def __init__(self, backbone_name, hidden_size, vocab_size):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Modules
        self.feat_to_hidden = nn.Linear(self.backbone.out_features, self.hidden_size, bias=False)
        self.rnn = nn.GRU(hidden_size,hidden_size, batch_first=True)

    def forward(self, images, embdedded_questions):
        img_feats = self.backbone(images)
        img_feats = self.feat_to_hidden(img_feats)
        # Concatenate image features and token embeddings
        x = torch.cat((img_feats.unsqueeze(1), embdedded_questions), dim=1)
        # Feed to GRU
        x, _ = self.rnn(x)

        return x[:,-1,:]
    
class BackBone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if(backbone=="resnet18"):
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.transforms = ResNet18_Weights.DEFAULT.transforms()
            self.out_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.transforms(x)
        return self.backbone(x).squeeze()


class VaE(nn.Module):
    '''
    Variational Autoencoder model that gets in input a batch of GRUs last hidden states and projects them in a VAE latent space.
    In training mode it samples from the vae latent distribution using the reparameterization trick.
    In inference mode it returns the mean of the projected latent distribution.
    '''
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # Modules
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size)

    def forward(self, x, mode="train"):
        # Project to latent space
        mean = self.hidden_to_mean(x)
        logvar = self.hidden_to_logvar(x)
        # Sample from latent distribution
        if(mode=="train"):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
        elif(mode=="inference"):
            z = mean
        else:
            raise NotImplementedError
        # Project back to hidden space
        x = self.latent_to_hidden(z)
        return x, mean, logvar


class CreativityDecoder(nn.Module):
    '''
    Decoder for creativity model.
    It takes in a batch of images, a batch of latent representations, and a batch of questions. 
    It extracts image features from the image using a backbone.
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    '''

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        