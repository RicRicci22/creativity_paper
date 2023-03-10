import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# For beam_decoding
import operator
from queue import PriorityQueue
from utils import BeamSearchNode, cross_entropy_loss, kl_loss


class Im2QModel(BaseModel):
    """
    Basic model which extracts features from the images, and produces for each image a single question
    """

    def __init__(
        self,
        backbone_name,
        hidden_size,
        vocab_size,
        dropout,
        sos_token,
        eos_token,
        concatenate=True,
        device="cpu",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name, hidden_size, freeze=True)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device
        self.decoder = Im2QDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dropout=dropout,
            sos_token=sos_token,
            device=device,
            concatenate=concatenate,
        )
        self.concatenate = concatenate
        self.dropout = nn.Dropout(dropout)
        self.hiddens_to_logits_w = nn.Parameter(
            torch.randn(self.hidden_size, self.vocab_size, device=self.device) * 0.001
        )  # Initialize at a low number
        self.hiddens_to_logits_b = nn.Parameter(
            torch.randn(self.vocab_size, device=self.device) * 0
        )  # Initialize at zero

    def forward(self, images, questions, lenghts):
        # Embed the questions
        img_feats = self.dropout(self.backbone(images))
        # Forward to the decoder
        decoder_hiddens = self.decoder(img_feats, questions, lenghts)
        logits = (
            self.dropout(decoder_hiddens) @ self.hiddens_to_logits_w
            + self.hiddens_to_logits_b
        )
        return logits

    def sample(self, images, max_len=50, multinomial=True):
        self.eval()
        with torch.no_grad():
            # Extract image features
            img_feats = self.backbone(images)
            sos_token = self.decoder.embedding(
                torch.tensor(
                    [self.sos_token], dtype=torch.long, device=self.device
                ).expand(img_feats.shape[0], 1)
            )
            if self.concatenate:
                # Concatenate image features and sos token in the 1 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=1)
            else:
                # Concatenate image features and sos token in the 2 dimension
                x = torch.cat((img_feats.unsqueeze(1), sos_token), dim=2)

            # Start decoding by feeding the GRU
            states = None
            args = range(images.shape[0])
            output = torch.empty(
                (images.shape[0], 0), device=self.device, dtype=torch.long
            )
            for _ in range(max_len):
                x = self.decoder.embed_ln(x)
                hiddens, states = self.decoder.rnn(x, states)
                hiddens = self.decoder.post_ln(hiddens)
                logits = hiddens @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
                # Get the most likely token
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=1)
                if multinomial:
                    predicted = torch.multinomial(probs, num_samples=1)
                else:
                    predicted = torch.argmax(probs, dim=1).unsqueeze(1)
                output = torch.cat((output, predicted), dim=1)
                # Get the embedding of the predicted token
                if self.concatenate:
                    x = self.decoder.embedding(predicted)
                else:
                    x = torch.cat(
                        (img_feats.unsqueeze(1), self.decoder.embedding(predicted)),
                        dim=2,
                    )
                # If the predicted token is the eos token, stop decoding
                args = list(
                    set(args)
                    - set(torch.where(predicted == self.eos_token)[0].tolist())
                )
                if len(args) == 0:
                    break

        self.train()
        return output.clone().detach().requires_grad_(False)

    def beam_decode(self, images, max_len=50, beam_width=10, topk=1):
        """
        :param images: images matrix tensor of shape [B, H] where B is the batch size, H is the features hidden size
        :param max_len: maximum length of the generated output sequence
        :param beam_width: number of sequences to keep track of at each step
        :param topk: number of top probable sequences to return at the end
        :return: decoded_batch
        """

        decoded_batch = []
        img_feats = self.backbone(images)
        sos_token = self.decoder.embedding(
            torch.tensor([[self.sos_token]], dtype=torch.long, device=self.device)
        )
        # decoding goes sentence by sentence
        for idx in range(img_feats.size(0)):
            # Start with the start of the sentence token

            if self.concatenate:
                # Concatenate image features and sos token in the 1 dimension
                decoder_input = torch.cat(
                    (
                        img_feats[idx, :].unsqueeze(0).unsqueeze(1),
                        sos_token,
                    ),
                    dim=1,
                )
            else:
                # Concatenate image features and sos token in the 2 dimension
                decoder_input = torch.cat(
                    (
                        img_feats[idx, :].unsqueeze(0).unsqueeze(1),
                        sos_token,
                    ),
                    dim=2,
                )

            decoder_input = self.decoder.embed_ln(decoder_input)

            # Number of sentence to generate
            endnodes = []

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_input, None, None, self.sos_token, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                score, n = nodes.get()

                while n.wordid == self.eos_token and n.prevNode != None:
                    endnodes.append((score, n))
                    score, n = nodes.get()
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= topk:
                        break

                if len(endnodes) >= topk:
                    break

                decoder_input = n.input
                decoder_hidden = n.h

                decoder_input = self.decoder.embed_ln(decoder_input)

                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder.rnn(
                    decoder_input, decoder_hidden
                )
                decoder_output = self.decoder.post_ln(decoder_output)
                # PUT HERE REAL BEAM SEARCH OF TOP
                # Hiddens (batch_size, 1, hidden_size)
                logits = (
                    decoder_output @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
                )  # (batch_size, 1, vocab_size)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=1)

                indexes = torch.multinomial(
                    probs, num_samples=beam_width, replacement=False
                )
                probs = probs.gather(1, indexes)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0, new_k]
                    p = probs[0][new_k].item()
                    next_input = self.decoder.embedding(
                        torch.tensor(
                            [[decoded_t]], dtype=torch.long, device=self.device
                        )
                    )
                    if self.concatenate:
                        # Concatenate image features and sos token in the 1 dimension
                        decoder_input = next_input
                    else:
                        # Concatenate image features and sos token in the 2 dimension
                        decoder_input = torch.cat(
                            (img_feats[idx, :].unsqueeze(0).unsqueeze(1), next_input),
                            dim=2,
                        )

                    node = BeamSearchNode(
                        decoder_input,
                        decoder_hidden,
                        n,
                        decoded_t,
                        (n.cumul_prob * (n.len - 1) + p) / n.len,
                        n.len + 1,
                    )
                    score = node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            # print("SORTED")
            # print(sorted(endnodes, key=operator.itemgetter(0)))
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                cumul_prob = n.cumul_prob
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append((cumul_prob, utterance))

            decoded_batch.append(utterances)

        return decoded_batch


class Im2QDecoder(nn.Module):
    """
    Simple decoder which takes image features and produce a question for each image.
    """

    def __init__(
        self,
        hidden_size,
        dropout,
        vocab_size,
        sos_token,
        concatenate=True,
        device="cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if concatenate:
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.embed_ln = nn.LayerNorm(hidden_size)
        else:
            self.rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
            self.embed_ln = nn.LayerNorm(hidden_size * 2)
        self.concatenate = concatenate
        self.dropout = nn.Dropout(dropout)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, img_features, questions, lenghts):
        # Embed the questions
        embedded = self.embedding(questions)
        if self.concatenate:
            # Concatenate the image features and the embedded questions
            input = torch.cat((img_features.unsqueeze(1), embedded), dim=1)
            input = self.embed_ln(input)
            # Pack the padded sequence
            packed = pack_padded_sequence(input, lenghts, batch_first=True)
        else:
            input = torch.cat(
                (img_features.unsqueeze(1).expand(-1, embedded.shape[1], -1), embedded),
                dim=2,
            )
            input = self.embed_ln(input)
            packed = pack_padded_sequence(
                input, [l - 1 for l in lenghts], batch_first=True
            )
        # Feed the packed sequence to the RNN
        packed_output, _ = self.rnn(packed)
        if self.concatenate:
            # Pad the packed sequence
            padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
            padded_output = padded_output[
                :, 1:, :
            ]  # Remove the first token which is the image feature
            # Pack again to be more efficient
            packed_output = pack_padded_sequence(
                padded_output, [l - 1 for l in lenghts], batch_first=True
            )

        # Not sure if this ln is good
        packed_output = self.post_ln(packed_output[0])

        return packed_output


class CreativityModel(BaseModel):
    def __init__(
        self,
        backbone_name,
        hidden_size,
        latent_size,
        vocab_size,
        sos_token,
        eos_token,
        only_image=False,
        device="cpu",
        dropout=0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = BackBone(backbone_name, hidden_size, freeze=True)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device
        self.only_image = only_image
        self.dropout = nn.Dropout(dropout)

        self.encoder = CreativityEncoder(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=device,
            only_image=only_image,
        )

        self.vae = VaE(hidden_size=hidden_size, latent_size=latent_size, device=device)
        self.decoder = CreativityDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            sos_token=sos_token,
            device=device,
        )

        self.hiddens_to_logits_w = nn.Parameter(
            torch.randn(self.hidden_size, self.vocab_size, device=self.device) * 0.001
        )  # Initialize at a low number
        self.hiddens_to_logits_b = nn.Parameter(
            torch.randn(self.vocab_size, device=self.device) * 0
        )  # Initialize at zero

    def forward(self, images, questions, lenghts, mode="train"):
        # Embed the questions
        img_feats = self.backbone(images)
        # Encode the images and questions

        hiddens = self.encoder(img_feats, questions, lenghts)

        hiddens = self.dropout(hiddens)

        # Project to VaE latent space
        latents, mus, logvars = self.vae(hiddens, mode=mode)
        # Forward all to the decoder
        decoder_hiddens = self.decoder(img_feats, latents, questions, lenghts)
        # Get logits from decoder hiddens
        logits = (
            self.dropout(decoder_hiddens) @ self.hiddens_to_logits_w
            + self.hiddens_to_logits_b
        )
        return logits, mus, logvars

    def sample(self, images, max_len=50, multinomial=True):
        self.eval()
        with torch.no_grad():
            # Sample from prior and decode a question for the image
            img_feats = self.backbone(images)
            if self.only_image:
                latents, _, _ = self.vae(img_feats, mode="train")
            else:
                # Sample from prior
                latents = self.vae.sample_prior(img_feats.shape[0])
            # Concatenate image features, latents and sos token
            sos_token = self.decoder.embedding(
                torch.tensor(
                    [self.sos_token], dtype=torch.long, device=self.device
                ).expand(img_feats.shape[0], 1)
            )
            x = torch.cat(
                (img_feats.unsqueeze(1), latents.unsqueeze(1), sos_token), dim=1
            )
            # Start decoding by feeding the GRU
            states = None
            args = range(images.shape[0])
            output = torch.empty(
                (images.shape[0], 0), device=self.device, dtype=torch.long
            )
            for _ in range(max_len):
                x = self.decoder.ln_embed(x)
                hiddens, states = self.decoder.rnn(x, states)
                hiddens = self.decoder.post_ln(hiddens)
                logits = hiddens @ self.hiddens_to_logits_w + self.hiddens_to_logits_b
                # Get the most likely token
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=1)
                if multinomial:
                    predicted = torch.multinomial(probs, num_samples=1)
                else:
                    predicted = torch.argmax(probs, dim=1).unsqueeze(1)
                # Add the predicted token to the output
                output = torch.cat((output, predicted), dim=1)
                # Get the embedding of the predicted token
                x = self.decoder.embedding(predicted)
                # If the predicted token is the eos token, stop decoding
                args = list(
                    set(args)
                    - set(torch.where(predicted == self.eos_token)[0].tolist())
                )
                if len(args) == 0:
                    break

        self.train()
        return output.clone().detach().requires_grad_(False)

    def get_loss(output, target, mean=None, logvar=None):
        if mean == None and logvar == None:
            return cross_entropy_loss(output, target)
        else:
            return cross_entropy_loss(output, target), kl_loss(mean, logvar)


class CreativityEncoder(nn.Module):

    """
    Encoder for creativity model.
    It takes in a batch of image features, and a batch of tokenized questions.
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    """

    def __init__(self, hidden_size, vocab_size, device, only_image=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.only_image = only_image
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.ln_embeddings = nn.LayerNorm(hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, img_feats, questions, lengths=None):
        if self.only_image:
            return self.ln_out(img_feats)
        else:
            embdedded_questions = self.embedding(questions)
            # Concatenate image features and token embeddings
            x = torch.cat(
                (img_feats.unsqueeze(1), embdedded_questions[:, 1:, :]), dim=1
            )
            # Apply layernorm
            x = self.ln_embeddings(x)
            if lengths is not None:
                x = pack_padded_sequence(x, [l - 1 for l in lengths], batch_first=True)
            # Feed to GRU
            x, _ = self.rnn(x)
            if lengths is not None:
                x, l = pad_packed_sequence(x, batch_first=True)
            # Apply layernorm
            x = self.ln_out(x)
            return x[
                range(img_feats.shape[0]), list(l - 1), :
            ]  # l is a lenght, so to convert to the index I need to substract 1


class BackBone(nn.Module):
    def __init__(self, backbone, hidden_size, freeze=True):
        super().__init__()
        if backbone == "resnet18":
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
        return self.to_hidden(x.squeeze(2).squeeze(2))


class VaE(nn.Module):
    """
    Variational Autoencoder model that gets in input a batch of GRUs last hidden states and projects them in a VAE latent space.
    In training mode it samples from the vae latent distribution using the reparameterization trick.
    In inference mode it returns the mean of the projected latent distribution.
    """

    def __init__(self, hidden_size, latent_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # Modules
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.hidden_to_logvar = nn.Linear(
            self.hidden_size, self.latent_size, bias=False
        )

        self.latent_to_hidden = nn.Linear(
            self.latent_size, self.hidden_size, bias=False
        )
        self.device = device
        self.ln_hiddens_out = nn.LayerNorm(hidden_size)

    def forward(self, x, mode="train"):
        # Project to latent space
        mean = self.hidden_to_mean(x)
        logvar = self.hidden_to_logvar(x)
        # Sample from latent distribution
        if mode == "train":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
        elif mode == "inference":
            z = mean
        else:
            raise NotImplementedError
        # Project back to hidden space
        x = self.ln_hiddens_out(self.latent_to_hidden(z))

        return x, mean, logvar

    def sample_prior(self, batch_size=1):
        # Sample from prior
        z = torch.randn(batch_size, self.latent_size).to(self.device)
        # Project back to hidden space
        x = self.ln_hiddens_out(self.latent_to_hidden(z))
        return x


class CreativityDecoder(nn.Module):
    """
    Decoder for creativity model.
    It takes in a batch of image features, a batch of latent representations, and a batch of tokenized questions.
    It concatenate token embeddings to the image features.
    It feeds all this to a GRU and returns the last hidden state of the GRU.
    """

    def __init__(self, hidden_size, vocab_size, sos_token, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        # Modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.device = device
        self.ln_embed = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, img_feats, latent_codes, questions, lenghts):
        # Questions contains both the start and the end tokens
        embdedded_questions = self.embedding(questions)
        # Concatenate image features, latents, and token embeddings
        x = torch.cat(
            (img_feats.unsqueeze(1), latent_codes.unsqueeze(1), embdedded_questions),
            dim=1,
        )
        x = self.ln_embed(x)
        # Pack the sequence
        x = pack_padded_sequence(x, [l + 1 for l in lenghts], batch_first=True)
        # Feed to GRU
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[:, 2:, :]  # Remove the first two tokens (img feats and latent token)
        # Pack again to be more efficient
        x = pack_padded_sequence(x, [l - 1 for l in lenghts], batch_first=True)
        x = self.post_ln(x[0])
        return x
