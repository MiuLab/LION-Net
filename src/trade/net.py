import torch
import torch.nn as nn

from modules.embedding import Embedding
from trade.classifier import Classifier
from trade.encoder import Encoder
from trade.decoder import Decoder
from trade.loader import Loader
from trade.gate import Gate


class Net(nn.Module):
    def __init__(self, config, vocab):
        super(Net, self).__init__()
        if config.emb.share_embed is False:
            self.enc_emb = self.sch_emb = Embedding(config, vocab)
            self.dec_emb = Embedding(config, vocab)
        else:
            emb = Embedding(config, vocab)
            self.enc_emb = self.dec_emb = self.sch_emb = emb
        config.emb.dim = self.enc_emb.embedding_dim
        self.enc = Encoder(config, self.enc_emb)
        self.dec = Decoder(config, vocab, self.dec_emb)
        self.sch = Loader(config, self.sch_emb)
        self.gat = Gate(config)
        self.act_cls = Classifier(config)
        self.req_cls = Classifier(config)

    def load_sch_embed(self, file_path, device):
        self.sch.load_embed(file_path, device)

    def forward(self, batch):
        enc_i = batch['context']
        enc_o = [self.enc(i) for i in enc_i]
        # active intent
        act_o = []
        if self.sch.sch_type == "rnn":
            service_idx = batch['active_intent']['service_desc']
            intent_idx = batch['active_intent']['intent_desc']
        else:
            service_idx = batch['active_intent']['service_idx']
            intent_idx = batch['active_intent']['intent_idx']
        for out, idx_0, idx_1 in zip(
                enc_o,
                service_idx,
                intent_idx):
            sch_o = self.sch(idx_0, idx_1, 'int')
            cls_o = self.act_cls(out[1], sch_o)
            act_o.append(cls_o)
        # requested slots
        req_o = []
        if self.sch.sch_type == "rnn":
            service_idx = batch['requested_slots']['service_desc']
            slot_idx = batch['requested_slots']['slot_desc']
        else:
            service_idx = batch['requested_slots']['service_idx']
            slot_idx = batch['requested_slots']['slot_idx']
        for out, idx_0, idx_1 in zip(
                enc_o,
                service_idx,
                slot_idx):
            sch_o = self.sch(idx_0, idx_1, 'slt')
            cls_o = self.req_cls(out[1], sch_o)
            req_o.append(cls_o)
        # slot filling
        dec_o = []
        max_len = batch['slot_filling']['max_len']
        if self.sch.sch_type == "rnn":
            service_idx = batch['slot_filling']['value_service_desc']
            slot_idx = batch['slot_filling']['value_slot_desc']
        else:
            service_idx = batch['slot_filling']['value_service_idx']
            slot_idx = batch['slot_filling']['value_slot_idx']
        for out, idx_0, idx_1, ext_l, ext_i, unk_v, cat_f in zip(
                enc_o,
                service_idx,
                slot_idx,
                batch['ext_list'],
                batch['ext_context'],
                batch['slot_filling']['value_idx'],
                batch['slot_filling']['is_categorical']):
            sch_o = self.sch(idx_0, idx_1, 'slt')
            ext_z = torch.zeros(len(ext_l))
            ext_z = ext_z.to(device=sch_o.device)
            val_o = self.dec(
                enc_o=out,
                sch_o=sch_o,
                ext_z=ext_z,
                ext_i=ext_i,
                unk_v=unk_v,
                cat_f=cat_f,
                max_len=max_len)
            dec_o.append(val_o)
        # context gate
        cxt_o = []
        if self.sch.sch_type == "rnn":
            service_idx = batch['slot_filling']['context_service_desc']
            slot_idx = batch['slot_filling']['context_slot_desc']
        else:
            service_idx = batch['slot_filling']['context_service_idx']
            slot_idx = batch['slot_filling']['context_slot_idx']
        for out, idx_0, idx_1 in zip(
                enc_o,
                service_idx,
                slot_idx):
            sch_o = self.sch(idx_0, idx_1, 'slt')
            gat_o = self.gat(out[1], sch_o)
            cxt_o.append(gat_o)

        return act_o, req_o, dec_o, cxt_o

