import numpy as np
import sys
import torch
import torch.nn as nn

from modules.attn import Attn


class Decoder(nn.Module):
    def __init__(self, config, vocab, emb):
        super(Decoder, self).__init__()
        self.type = getattr(config, "rnn", "GRU")
        self.dim = config.dec.dim
        self.vocab_size = emb.num_embeddings
        self.force_copy = config.dec.force_copy
        self.bos = vocab.convert_tokens_to_indices(["<BOS>"])[0]
        self.eos = vocab.convert_tokens_to_indices(["<EOS>"])[0]
        self.unk = vocab.convert_tokens_to_indices(["<UNK>"])[0]
        self.pad = vocab.convert_tokens_to_indices(["<PAD>"])[0]
        self.dontcare = vocab.convert_tokens_to_indices(["<DONTCARE>"])[0]

        self.special_tokens = \
            [self.bos, self.eos, self.unk, self.pad, self.dontcare]

        self.emb = emb
        self.rnn = getattr(torch.nn, self.type)(
                input_size=config.emb.dim,
                hidden_size=config.dec.dim,
                num_layers=config.dec.lyr,
                dropout=(config.dec.drp if config.dec.lyr > 1 else 0),
                batch_first=True)
        self.h_proj = nn.Linear(
                config.enc.dim * (1 + config.enc.bid) * config.enc.lyr,
                config.dec.dim * config.dec.lyr)
        self.hs_proj = nn.Linear(
                config.sch.dim,
                config.dec.dim * config.dec.lyr)
        if self.type == "GRU":
            self.attn = Attn(
                method=getattr(config.dec, "attn", "none"),
                k_dim=config.enc.dim * (1 + config.enc.bid),
                v_dim=config.enc.dim * (1 + config.enc.bid),
                q_dim=config.dec.dim * config.dec.lyr)
            self.gate = nn.Linear(
                config.dec.dim + config.emb.dim + self.attn.o_dim, 1)
        elif self.type == "LSTM":
            self.c_proj = nn.Linear(
                config.enc.dim * (1 + config.enc.bid) * config.enc.lyr,
                config.dec.dim * config.dec.lyr)
            self.cs_proj = nn.Linear(
                    config.sch.dim,
                    config.dec.dim * config.dec.lyr)
            self.attn = Attn(
                method=getattr(config.dec, "attn", "none"),
                k_dim=config.enc.dim * (1 + config.enc.bid),
                v_dim=config.enc.dim * (1 + config.enc.bid),
                q_dim=config.dec.dim * config.dec.lyr * 2,
                device=self.device)
            self.gate = nn.Linear(
                (config.dec_dim * 2) + config.emb.dim + self.attn.o_dim, 1)
        self.proj1 = nn.Linear(
            config.dec.dim + self.attn.o_dim, config.dec.dim)
        self.proj2 = nn.Linear(
            config.dec.dim, self.emb.num_embeddings)

    def forward(
            self,
            enc_o, sch_o,
            ext_z, ext_i, unk_v,
            cat_f, max_len=10):
        logits = []
        if self.type == "GRU":
            enc_o, enc_h = enc_o
            enc_o = enc_o.repeat(sch_o.size(0), 1, 1)
            enc_h = enc_h.repeat(1, sch_o.size(0), 1)
            dec_h = self._proj(sch_o, enc_h)
        elif self.type == "LSTM":
            enc_o, (enc_h, enc_c) = enc_o
            enc_o = enc_o.repeat(sch_o.size(0), 1, 1)
            enc_h = enc_h.repeat(1, sch_o.size(0), 1)
            enc_c = enc_c.repeat(1, sch_o.size(0), 1)
            dec_h, dec_c = self._proj(sch_o, enc_h, enc_c)
        ext_z = ext_z.repeat(sch_o.size(0), 1)
        ext_i = ext_i.repeat(sch_o.size(0), 1)

        for idx in range(max_len):
            if idx == 0:
                x = torch.full(
                    (enc_o.size(0), 1), self.bos,
                    dtype=torch.long, device=enc_o.device)
            else:
                if unk_v is not None:
                    x = unk_v[:, idx-1].unsqueeze(1)
                else:
                    x = last_x.masked_fill(
                        (last_x >= self.vocab_size), self.unk)

            dec_i = self.emb(x)
            if self.type == "GRU":
                dec_r, dec_h = self.rnn(dec_i, dec_h)
                flat_dec_h = dec_h.permute(1, 0, 2).reshape(dec_h.size(1), -1)
                attn_i = flat_dec_h
            elif self.type == "LSTM":
                dec_r, (dec_h, dec_c) = self.rnn(dec_i, (dec_h, dec_c))
                flat_dec_h = dec_h.permute(1, 0, 2).reshape(dec_h.size(1), -1)
                flat_dec_c = dec_c.permute(1, 0, 2).reshape(dec_c.size(1), -1)
                attn_i = torch.cat([flat_dec_h, flat_dec_c], dim=-1)

            dist, attn = self.attn(enc_o, enc_o, attn_i)

            proj_i = torch.cat([dec_r, attn], dim=-1)
            dec_o_vocab = self.proj2(self.proj1(proj_i))
            dec_o_vocab = torch.softmax(dec_o_vocab, dim=-1).squeeze(1)
            dec_o_vocab = torch.cat([dec_o_vocab, ext_z], dim=-1)
            gate_i = torch.cat([attn, attn_i.unsqueeze(1), dec_i], dim=-1)
            gate_o = torch.sigmoid(self.gate(gate_i)).squeeze(1)
            dec_o_vocab = dec_o_vocab * gate_o
            dec_o_copy = dist * (1 - gate_o)
            if self.force_copy:
                dec_o = []
                for sidx, slot in enumerate(ext_i):
                    i = ext_i[sidx].unsqueeze(0)
                    c = dec_o_copy[sidx].unsqueeze(0)
                    if cat_f[sidx]:
                        o = dec_o_vocab[sidx].unsqueeze(0)
                    else:
                        m = np.ones((self.vocab_size + ext_z.size(1)))
                        for t in self.special_tokens:
                            m[t] = 0
                        m = torch.BoolTensor(m).to(dec_o_copy.device)
                        o = dec_o_vocab[sidx].masked_fill_(m, 0).unsqueeze(0)
                    dec_o.append(o.scatter_add(1, i, c))
                dec_o = torch.stack(dec_o, dim=0).squeeze(1)
            else:
                dec_o = dec_o_vocab.scatter_add(1, ext_i, dec_o_copy)

            _, last_x = dec_o.max(dim=-1)
            last_x = last_x.unsqueeze(1)
            logits.append(dec_o)

        return torch.stack(logits, dim=1)

    def _proj(self, sch_o, enc_h, enc_c=None):
        bs = enc_h.size(1)
        flat_enc_h = enc_h.permute(1, 0, 2).reshape(bs, -1)
        proj_enc_h = self.h_proj(flat_enc_h)
        dec_h = torch.relu(proj_enc_h.reshape(bs, -1, self.dim))
        sch_h = torch.relu(
            self.hs_proj(sch_o).reshape(bs, dec_h.size(1), self.dim))
        dec_h = (dec_h + sch_h).permute(1, 0, 2)
        dec_h = dec_h.contiguous()
        if enc_c is not None:
            flat_enc_c = enc_c.permute(1, 0, 2).reshape(bs, -1)
            proj_enc_c = self.c_proj(flat_enc_c)
            dec_c = torch.relu(proj_enc_c.reshape(bs, -1, self.dim))
            sch_c = torch.relu(
                self.cs_proj(sch_o).reshape(bs, dec_c.size(1), self.dim))
            dec_c = (dec_c + sch_c).permute(1, 0, 2)
            dec_c = dec_c.contiguous()
            return dec_h, dec_c
        else:
            return dec_h
