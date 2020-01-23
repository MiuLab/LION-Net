import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, config):
        super(Gate, self).__init__()
        self.rnn_type = getattr(config, "rnn", "GRU")
        self.layers = nn.ModuleList()
        i_dim = (
            config.enc.dim * config.enc.lyr * (1 + config.enc.bid) +
            config.sch.dim)
        o_dim = config.cls.dim
        for _ in range(config.cls.lyr):
            self.layers.append(nn.Linear(i_dim, o_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=config.cls.drp))
            i_dim = config.cls.dim
        self.layers.append(
            nn.Linear(config.cls.dim, 2 + config.dc_gate))

    def forward(self, enc_h, sch_o):
        if self.rnn_type == "LSTM":
            enc_h = enc_h[0]
        flat_enc_h = enc_h.permute(1, 0, 2).reshape(enc_h.size(1), -1)
        flat_enc_h = flat_enc_h.repeat(sch_o.size(0), 1)
        cls_o = torch.cat([flat_enc_h, sch_o], dim=1)
        for layer in self.layers:
            cls_o = layer(cls_o)
        return cls_o
