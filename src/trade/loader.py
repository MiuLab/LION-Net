import pickle
import torch
import torch.nn as nn

from pathlib import Path

from modules.dataset import padding


class Loader(nn.Module):
    def __init__(self, config, emb):
        super(Loader, self).__init__()
        self.sch_type = config.sch.type
        self.use_ser = config.sch.service
        if self.sch_type == "rnn":
            self.type = getattr(config, "rnn", "GRU")
            self.slt_rnn = getattr(torch.nn, self.type)(
                input_size=config.emb.dim,
                hidden_size=config.sch.dim,
                num_layers=config.sch.lyr,
                dropout=(config.sch.drp if config.sch.lyr > 1 else 0),
                bidirectional=config.sch.bid,
                batch_first=True)
            self.int_rnn = getattr(torch.nn, self.type)(
                input_size=config.emb.dim,
                hidden_size=config.sch.dim,
                num_layers=config.sch.lyr,
                dropout=(config.sch.drp if config.sch.lyr > 1 else 0),
                bidirectional=config.sch.bid,
                batch_first=True)
            if self.use_ser:
                self.ser_rnn = getattr(torch.nn, self.type)(
                    input_size=config.emb.dim,
                    hidden_size=config.sch.dim,
                    num_layers=config.sch.lyr,
                    dropout=(config.sch.drp if config.sch.lyr > 1 else 0),
                    bidirectional=config.sch.bid,
                    batch_first=True)
                input_dim = (
                    config.sch.dim * (1 + config.sch.bid) *
                    config.sch.lyr * 2)
                self.slt_proj = nn.Linear(input_dim, config.sch.dim)
                self.int_proj = nn.Linear(input_dim, config.sch.dim)
            else:
                input_dim = (
                    config.sch.dim * (1 + config.sch.bid) * config.sch.lyr)
                self.slt_proj = nn.Linear(input_dim, config.sch.dim)
                self.int_proj = nn.Linear(input_dim, config.sch.dim)
            self.emb = emb
        else:
            self.ser_emb = None
            self.slt_emb = None
            self.int_emb = None
            if self.use_ser:
                self.slt_proj = nn.Linear(768 * 2, config.sch.dim)
                self.int_proj = nn.Linear(768 * 2, config.sch.dim)
            else:
                self.slt_proj = nn.Linear(768, config.sch.dim)
                self.int_proj = nn.Linear(768, config.sch.dim)

    def forward(self, idx_0, idx_1, mode):
        if self.sch_type == "rnn":
            emb_1 = self.emb(idx_1)
            if self.type == "GRU":
                if self.use_ser:
                    emb_0 = self.emb(idx_0)
                    _, h_0 = self.ser_rnn(emb_0)
                if mode == "slt":
                    _, h_1 = self.slt_rnn(emb_1)
                elif mode == "int":
                    _, h_1 = self.int_rnn(emb_1)
            elif self.type == "LSTM":
                if self.use_ser:
                    _, (h_0, _) = self.ser_rnn(emb_0)
                if mode == "slt":
                    _, (h_1, _) = self.slt_rnn(emb_1)
                elif mode == "int":
                    _, (h_1, _) = self.int_rnn(emb_1)
            if self.use_ser:
                h_0 = h_0.permute(1, 0, 2).reshape(h_0.size(1), -1)
                h_1 = h_1.permute(1, 0, 2).reshape(h_1.size(1), -1)
                h_1 = torch.cat([h_0, h_1], dim=1)
            else:
                h_1 = h_1.permute(1, 0, 2).reshape(h_1.size(1), -1)
            if mode == "slt":
                out = self.slt_proj(h_1)
            elif mode == "int":
                out = self.int_proj(h_1)
        elif self.sch_type == "embed":
            if self.use_ser:
                emb_0 = self.ser_emb(idx_0)
            if mode == "slt":
                emb_1 = self.slt_emb(idx_1)
            elif mode == "int":
                emb_1 = self.int_emb(idx_1)
            if self.use_ser:
                emb_1 = torch.cat([emb_0, emb_1], dim=1)
            if mode == "slt":
                out = self.slt_proj(emb_1)
            elif mode == "int":
                out = self.int_proj(emb_1)
        return out

    def load_embed(self, file_path, device):
        ser_weight, int_weight, slt_weight = \
            pickle.load(open(file_path, 'rb'))
        self.ser_emb = nn.Embedding.from_pretrained(
            ser_weight, freeze=True)
        self.int_emb = nn.Embedding.from_pretrained(
            int_weight, freeze=True)
        self.slt_emb = nn.Embedding.from_pretrained(
            slt_weight, freeze=True)
        self.ser_emb.to(device=device)
        self.int_emb.to(device=device)
        self.slt_emb.to(device=device)
