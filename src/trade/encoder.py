import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config, emb):
        super(Encoder, self).__init__()
        self.type = getattr(config, "rnn", "GRU")
        self.rnn = getattr(torch.nn, self.type)(
            input_size=config.emb.dim,
            hidden_size=config.enc.dim,
            num_layers=config.enc.lyr,
            dropout=(config.enc.drp if config.enc.lyr > 1 else 0),
            bidirectional=config.enc.bid,
            batch_first=True)
        self.emb = emb

    def forward(self, cxt):
        emb_o = self.emb(cxt)
        enc_o, enc_h = self.rnn(emb_o)
        return enc_o, enc_h
