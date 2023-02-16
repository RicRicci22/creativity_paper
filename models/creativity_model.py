import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Im2QModel(BaseModel):
    '''
    Basic model which extracts features from the images, and produces for each image a single question
    '''
    def __init__(self, backbone_name, hidden_size, vocab_size, sos_token, eos_token, concatenate = True, device = 'cpu'):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name, hidden_size, freeze=True)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token  = eos_token
        self.device = device
        self.decoder = Im2QDecoder(hidden_size, vocab_size, sos_token, device = device, concatenate = concatenate)
        self.concatenate = concatenate
        self.hiddens_to_logits_w = nn.Parameter(torch.randn(self.hidden_size, self.vocab_size, device=self.device) * 0.001) # Initialize at a low number 
        self.hiddens_to_logits_b = nn.Parameter(torch.randn(self.vocab_size, device=self.device) * 0) # Initialize at zero
            
    def forward(self, images, questions, lenghts):
        # Embed the questions
        img_feats = self.backbone(images)
        # Forward to the decoder
        decoder_hiddens = self.decoder(img_feats, questions, lenghts)
        logits = decoder_hiddens[0] @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
        return logits
    
    def sample(self, images, max_len=50):
        self.eval()
        with torch.no_grad():
            # Extract image features
            img_feats = self.backbone(images)
            sos_token = self.decoder.embedding(torch.tensor([self.sos_token],dtype = torch.long, device=self.device).expand(img_feats.shape[0], 1))
            if(self.concatenate):
                # Concatenate image features and sos token in the 1 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=1)
            else: 
                # Concatenate image features and sos token in the 2 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=2)
                
            # Start decoding by feeding the GRU
            states = None
            args = range(images.shape[0])
            output = torch.empty((images.shape[0], 0), device=self.device, dtype=torch.long)
            for _ in range(max_len):
                hiddens, states = self.decoder.rnn(x, states)
                logits = hiddens @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
                # Get the most likely token
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim=1)
                predicted = torch.multinomial(probs, num_samples=1)
                output = torch.cat((output, predicted), dim=1)
                # Get the embedding of the predicted token
                if(self.concatenate):
                    x = self.decoder.embedding(predicted)
                else:
                    x = torch.cat((img_feats.unsqueeze(1), self.decoder.embedding(predicted)), dim=2)
                # If the predicted token is the eos token, stop decoding
                args = list(set(args)-set(torch.where(predicted==self.eos_token)[0].tolist()))
                if len(args) == 0:
                    break

        self.train()
        return output.clone().detach().requires_grad_(False)
    
    def beam_search(self, images, max_len = 50, k = 1):
        '''
        Beam search for the model, generates k of the most probable questions for every input image. 
        In general it expects that images have a batch dimension (it can be also a single image)
        # Arguments:
        images: Tensor of shape (batch_size, channels, height, width)
        max_len: Maximum length of the generated questions
        k: Number of the most probable questions to generate for each image
        '''
        self.eval()
        with torch.no_grad():
            # Extract image features
            img_feats = self.backbone(images) # (batch_size, hidden_size)
            sos_token = self.decoder.embedding(torch.tensor([self.sos_token],dtype = torch.long, device=self.device).expand(img_feats.shape[0], 1))
            if(self.concatenate):
                # Concatenate image features and sos token in the 1 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=1)
            else: 
                # Concatenate image features and sos token in the 2 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=2)
            
            # Start decoding by feeding the GRU
            states = None
            # Initialize at zero the tensor containing the cumulative probabilities of the generated sequences
            cumul_probs = torch.zeros((images.shape[0], k*k), device=self.device) # (batch_size, k*k)
            output = torch.empty((images.shape[0], k), device=self.device, dtype=torch.long) # (batch_size, k*k)
            for i in range(max_len):
                hiddens, states = self.decoder.rnn(x, states) 
                if (i==0):
                    # Hiddens (batch_size, 1, hidden_size)
                    logits = hiddens @ self.hiddens_to_logits_w + self.hiddens_to_logits_b # (batch_size, 1, vocab_size)
                    logits = logits[:,-1,:]
                    probs = F.softmax(logits, dim=1)
                    predicted = torch.multinomial(probs, num_samples=k) # (batch_size, k)
                    cumul_probs += torch.repeat_interleave(probs, k, dim=1) # (batch_size, k*k)
                    output = torch.cat((output, predicted), dim=1)
                    # Get the embedding of the predicted token
                    if(self.concatenate):
                        x = self.decoder.embedding(predicted) # (batch_size, k, hidden_size)
                    else:
                        x = torch.cat((img_feats.unsqueeze(1), self.decoder.embedding(predicted)), dim=2)
            
        

class Im2QDecoder(nn.Module):
    '''
    Simple decoder which takes image features and produce a question for each image. 
    '''
    def __init__(self,hidden_size, vocab_size, sos_token, concatenate = True, device = "cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if(concatenate):
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size*2, hidden_size, batch_first=True)
        self.concatenate = concatenate
    
    def forward(self, img_features, questions, lenghts):
        # Embed the questions
        embedded = self.embedding(questions)
        if(self.concatenate):
            # Concatenate the image features and the embedded questions
            input = torch.cat((img_features.unsqueeze(1), embedded), dim=1)
            # Pack the padded sequence
            packed = pack_padded_sequence(input, lenghts, batch_first=True)
        else:
            input = torch.cat((img_features.unsqueeze(1).expand(-1,embedded.shape[1],-1), embedded), dim=2)
            packed = pack_padded_sequence(input, [l-1 for l in lenghts], batch_first=True)
        # Feed the packed sequence to the RNN
        packed_output, _ = self.rnn(packed)
        if(self.concatenate):
            # Pad the packed sequence
            padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
            padded_output = padded_output[:,1:,:] # Remove the first token which is the image feature
            # Pack again to be more efficient
            packed_output = pack_padded_sequence(padded_output, [l-1 for l in lenghts], batch_first=True)
            
        return packed_output

class CreativityModel(BaseModel):
    def __init__(self, backbone_name, hidden_size, latent_size, vocab_size, sos_token, eos_token, only_image = False, device = "cpu"):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name, hidden_size, freeze=True)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device
        self.only_image = only_image

        if not only_image:
            self.encoder = CreativityEncoder(hidden_size=self.hidden_size, vocab_size=self.vocab_size, device=device)

        self.vae = VaE(hidden_size=hidden_size, latent_size=latent_size, device=device)
        self.decoder = CreativityDecoder(hidden_size=hidden_size, vocab_size=vocab_size, sos_token=sos_token, device=device)

        self.hiddens_to_logits_w = nn.Parameter(torch.randn(self.hidden_size, self.vocab_size, device=self.device) * 0.001) # Initialize at a low number 
        self.hiddens_to_logits_b = nn.Parameter(torch.randn(self.vocab_size, device=self.device) * 0) # Initialize at zero
    
    def forward(self, images, questions, lenghts, mode="train"):
        # Embed the questions
        img_feats = self.backbone(images)
        # Encode the images and questions
        if(self.only_image):
            hiddens = img_feats
        else:
            hiddens = self.encoder(img_feats, questions, lenghts)
        # Project to VaE latent space
        latents, mus, logvars = self.vae(hiddens, mode=mode)
        # Forward all to the decoder
        decoder_hiddens = self.decoder(img_feats, latents, questions, lenghts)
        # Get logits from decoder hiddens
        logits = decoder_hiddens[0] @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
        return logits, mus, logvars
    
    def sample(self, images, max_len=50):
        self.eval()
        with torch.no_grad():
            # Sample from prior and decode a question for the image
            img_feats = self.backbone(images)
            if(self.only_image):
                latents, _, _ = self.vae(img_feats, mode="train")
            else:
                # Sample from prior
                latents = self.vae.sample_prior(img_feats.shape[0])
            # Concatenate image features, latents and sos token
            sos_token = self.decoder.embedding(torch.tensor([self.sos_token],dtype = torch.long, device=self.device).expand(img_feats.shape[0], 1))
            x = torch.cat((img_feats.unsqueeze(1), latents.unsqueeze(1), sos_token), dim=1)
            # Start decoding by feeding the GRU
            states = None
            args = range(images.shape[0])
            output = torch.empty((images.shape[0], 0), device=self.device, dtype=torch.long)
            for _ in range(max_len):
                hiddens, states = self.decoder.rnn(x, states)
                logits = hiddens @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
                # Get the most likely token
                logits = logits[:,-1,:]
                probs = F.softmax(logits , dim=1)
                predicted = torch.multinomial(probs, num_samples=1)
                # Add the predicted token to the output
                output = torch.cat((output, predicted), dim=1)
                # Get the embedding of the predicted token
                x = self.decoder.embedding(predicted)
                # If the predicted token is the eos token, stop decoding
                args = list(set(args)-set(torch.where(predicted==self.eos_token)[0].tolist()))
                if len(args) == 0:
                    break

        self.train()
        return output.clone().detach().requires_grad_(False)

class CreativityEncoder(nn.Module):

    '''
    Encoder for creativity model.
    It takes in a batch of image features, and a batch of tokenized questions. 
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    '''

    def __init__(self, hidden_size, vocab_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size,hidden_size, batch_first=True)

    def forward(self, img_feats, questions, lengths=None):
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
    def __init__(self, backbone, hidden_size, freeze=True):
        super().__init__()
        if(backbone=="resnet18"):
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.transforms = torch.nn.Sequential(
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        )
            self.out_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.to_hidden = nn.Linear(self.out_features, hidden_size, bias=False)
        else:
            raise NotImplementedError
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.transforms(x)
        x = self.backbone(x)
        return self.to_hidden(x.squeeze())

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
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size, bias=False)
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

    def __init__(self, hidden_size, vocab_size, sos_token, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size,hidden_size, batch_first=True)
        self.device = device
    
    def forward(self, img_feats, latent_codes, questions, lenghts):
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
        
class OnlyImageEncoder(nn.Module):
    '''
    Encoder implemented as in https://ieeexplore.ieee.org/document/9671493. It basically consist of a backbone which takes a batch of images and returns a batch of image features.
    It optionally append on top of it a linear layer that projects the image features to a hidden space of size hidden_size.
    '''
    def __init__(self, backbone_name, hidden_size, freeze=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.freeze = freeze
        self.backbone = BackBone(backbone_name, hidden_size, freeze)
    
    def forward(self, x):
        x = self.backbone(x)
        return x