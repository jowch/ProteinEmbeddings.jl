import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import (TransformerEncoder, TransformerEncoderLayer,
                      TransformerDecoder, TransformerDecoderLayer)
from torch.nn.init import kaiming_normal_

class TransformerVAE(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.nhead = nhead
        self.ntoken = ntoken

        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.mu = nn.Linear(d_model, ntoken)
        self.log_var = nn.Linear(d_model, ntoken)

        self.latent_decoder = nn.Linear(ntoken, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, ntoken),
            nn.LogSoftmax(dim=1)
        )

        # self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                kaiming_normal_(p)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # print(f'input: {src.shape}')
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # print(f'pos. encoded: {src.shape}')

        tgt = self.encoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        embedding = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # print(f'embedded: {embedding.shape}')

        mu = self.mu(embedding)
        lv = self.log_var(embedding)
        z = self.reparameterize(mu, lv)
        
        memory = self.latent_decoder(z)
        # print(f'sampled: {memory.shape}')

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        # print(f'decoded: {output.shape}')
        output = self.decoder(output)
        # print(f'output: {output.shape}')

        return [output, mu, lv]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        See: AntixK/PyTorch-VAE
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def criterion(self, recon: Tensor, targets: Tensor, mu: Tensor, log_var: Tensor):
        ce = F.cross_entropy(recon.view(-1, self.ntoken), targets.reshape(-1))
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld = (-0.5 * torch.sum(log_var - torch.pow(mu, 2) - torch.exp(log_var) + 1, 1)).mean().squeeze()

        return ce + kld

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

