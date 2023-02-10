import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CreativityModel(BaseModel):
    def __init__(self, backbone_name, hidden_size, latent_size, vocab_size, sos_token, eos_token, device):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name, freeze=True)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.backbone_feats = self.backbone.out_features
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device

        self.encoder = CreativityEncoder(backbone_feats=self.backbone_feats, hidden_size=self.hidden_size, vocab_size=self.vocab_size, device=device)
        self.vae = VaE(hidden_size=hidden_size, latent_size=latent_size, device=device)
        self.decoder = CreativityDecoder(backbone_feats=self.backbone_feats, hidden_size=hidden_size, vocab_size=vocab_size, sos_token=sos_token, device=device)

        self.hiddens_to_logits = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    
    def forward(self, images, questions, lenghts, mode="train"):
        # Embed the questions
        img_feats = self.backbone(images)
        # Encode the images and questions
        hiddens = self.encoder(img_feats, questions, lenghts)
        # Project to VaE latent space
        latents, mus, logvars = self.vae(hiddens, mode=mode)
        # Forward all to the decoder
        decoder_hiddens = self.decoder(img_feats, latents, questions, lenghts)
        # Get logits from decoder hiddens
        logits = self.hiddens_to_logits(decoder_hiddens[0])

        return logits, mus, logvars
    
    def sample(self, images, max_len=50):
        self.eval()
        with torch.no_grad():
            # Sample from prior and decode a question for the image
            img_feats = self.backbone(images)
            # Sample from prior
            latents = self.vae.sample_prior(img_feats.shape[0])
            # Concatenate image features, latents and sos token
            sos_token = self.decoder.embedding(torch.tensor([self.sos_token],dtype = torch.long, device=self.device).expand(img_feats.shape[0], 1))
            x = torch.cat((img_feats.unsqueeze(1), latents.unsqueeze(1), sos_token), dim=1)
            # Start decoding by feeding the GRU
            states = None
            args = range(images.shape[0])
            output = torch.empty((images.shape[0], 0), device=self.device)
            for _ in range(max_len):
                hiddens, states = self.decoder.rnn(x, states)
                logits = self.hiddens_to_logits(hiddens)
                # Get the most likely token
                predicted = torch.argmax(logits[:, -1, :], dim=1)
                output = torch.cat((output, predicted.unsqueeze(1)), dim=1)
                # Get the embedding of the predicted token
                x = self.decoder.embedding(predicted).unsqueeze(1)
                # If the predicted token is the eos token, stop decoding
                args = list(set(args)-set(torch.where(predicted==self.eos_token)[0].tolist()))
                if len(args) == 0:
                    break

        self.train()
        return output
        
    

class CreativityEncoder(nn.Module):

    '''
    Encoder for creativity model.
    It takes in a batch of image features, and a batch of tokenized questions. 
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    '''

    def __init__(self, backbone_feats, hidden_size, vocab_size, device):
        super().__init__()
        self.backbone_feats = backbone_feats
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.feat_to_hidden = nn.Linear(self.backbone_feats, self.hidden_size, bias=False)
        self.rnn = nn.GRU(hidden_size,hidden_size, batch_first=True)

    def forward(self, img_feats, questions, lengths=None):
        img_feats = self.feat_to_hidden(img_feats)
        embdedded_questions = self.embedding(questions)
        # Concatenate image features and token embeddings
        x = torch.cat((img_feats.unsqueeze(1), embdedded_questions[:,1:,:]), dim=1)
        if(lengths is not None):
            x = pack_padded_sequence(x, [l-1 for l in lengths], batch_first=True)
        # Feed to GRU
        x, _ = self.rnn(x)
        if(lengths is not None):
            x, l = pad_packed_sequence(x, batch_first=True)

        return x[range(img_feats.shape[0]),list(l-1),:] # l is a lenght, so to convert to the index I need to substract 1
    
class BackBone(nn.Module):
    def __init__(self, backbone, freeze=True):
        super().__init__()
        if(backbone=="resnet18"):
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.transforms = ResNet18_Weights.DEFAULT.transforms()
            self.out_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise NotImplementedError
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.transforms(x)
        return self.backbone(x).squeeze()


class VaE(nn.Module):
    '''
    Variational Autoencoder model that gets in input a batch of GRUs last hidden states and projects them in a VAE latent space.
    In training mode it samples from the vae latent distribution using the reparameterization trick.
    In inference mode it returns the mean of the projected latent distribution.
    '''
    def __init__(self, hidden_size, latent_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # Modules
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.device = device

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
    
    def sample_prior(self, batch_size=1):
        # Sample from prior
        z = torch.randn(batch_size, self.latent_size).to(self.device)
        # Project back to hidden space
        x = self.latent_to_hidden(z)
        return x


class CreativityDecoder(nn.Module):
    '''
    Decoder for creativity model.
    It takes in a batch of image features, a batch of latent representations, and a batch of tokenized questions. 
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    '''

    def __init__(self, backbone_feats, hidden_size, vocab_size, sos_token, device):
        super().__init__()
        self.backbone_feats = backbone_feats
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.feat_to_hidden = nn.Linear(self.backbone_feats, self.hidden_size, bias=False)
        self.rnn = nn.GRU(hidden_size,hidden_size, batch_first=True)
        self.device = device
    
    def forward(self, img_feats, latent_codes, questions, lenghts):
        # Project image features to hidden space
        img_feats = self.feat_to_hidden(img_feats)
        # Questions contains both the start and the end tokens
        embdedded_questions = self.embedding(questions)
        # Concatenate image features, latents, and token embeddings
        x = torch.cat((img_feats.unsqueeze(1), latent_codes.unsqueeze(1), embdedded_questions), dim=1)
        # Pack the sequence
        x = pack_padded_sequence(x, [l+1 for l in lenghts], batch_first=True)
        # Feed to GRU
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[:,2:,:] # Remove the first two tokens (img feats and latent token)
        # Pack again to be more efficient
        x = pack_padded_sequence(x, [l-1 for l in lenghts], batch_first=True)
        return x
        