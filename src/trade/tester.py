import copy
import csv
import json
import numpy as np
import os
import pickle
import re
import spacy
import torch
import torch.nn as nn

from numpy import linalg as LA
from pathlib import Path
from time import gmtime, strftime
from tqdm import tqdm

from modules.dataset import create_data_loader, transfer
from modules.logger import create_logger
from modules.metrics import compute_active_intent_acc
from modules.metrics import compute_requested_slots_f1
from modules.metrics import compute_slot_filling_acc
from modules.utils import create_device, extract_cat_slots, get_num_lines

from trade.model import Model
from trade.utils import extract_values


class Tester:
    def __init__(self, config, device):
        for k, v in config.test.items():
            setattr(self, k, v)
        self.dc_gate = config.model_param.dc_gate
        self.multi_value = config.train.multi_value
        self.sch_embed = (config.model_param.sch.type == "embed")

        nlp = spacy.load('en')
        self.tokenizer = \
            spacy.lang.en.English().Defaults().create_tokenizer(nlp)

        self.logger = create_logger(name="TEST")

        self.origin_dir = Path(config.data.data_dir)
        self.data_dir = Path(config.data.save_dir)
        self.exp_dir = self.origin_dir / "exp" / config.model / self.exp
        self.pred_dir = self.origin_dir / "prediction"
        if not self.pred_dir.exists():
            self.pred_dir.mkdir()

        self.config = config
        self.device = create_device(device)

        self.vocab = pickle.load(open(self.data_dir / "vocab.pkl", 'rb'))
        self.model = Model(
            config=config.model_param,
            vocab=self.vocab,
            device=self.device)
        self.logger.info(f"[-] Reading word vector......")
        self.emb = {}
        with open(config.data.embed_path, 'r') as file:
            for line in tqdm(file,
                             total=get_num_lines(config.data.embed_path),
                             leave=False):
                data = line.strip().split(' ')
                token, emb = data[0], list(map(float, data[1:]))
                self.emb[token] = emb

        if hasattr(self, "model_path"):
            self.model.load_state(
                self.model_path,
                save_device=config.train.device,
                load_device=config.test.device)
        else:
            self.model.load_best_state(
                self.exp_dir / "ckpt",
                save_device=config.train.device,
                load_device=config.test.device)

        self.trim_front = [',', '.', '?', '!', ':', "'"]
        self.trim_back = ['#']

    def test(self):
        test_files = list((self.origin_dir / "test").glob("dialogues_*.json"))
        test_files.sort()
        out_files = [
                self.pred_dir / f"dialogues_{idx+1:0>3}.json"
                for idx in range(len(test_files))]

        self.model.eval()
        preds = self.run_epoch(0, "test")

        count = 0
        for o, file_name in enumerate(tqdm(test_files)):
            test_dialogues = json.load(open(file_name))
            for d, dialogue in enumerate(test_dialogues):
                for t, turn in enumerate(dialogue['turns']):
                    if turn['speaker'] == "USER":
                        for f, frame in enumerate(turn['frames']):
                            state = {}
                            state['active_intent'] = \
                                preds['act_preds'][count]
                            state['requested_slots'] = \
                                preds['req_preds'][count]
                            state['slot_values'] = \
                                preds['slot_value_preds'][count]
                            count += 1
                            turn['frames'][f]['state'] = state
                        test_dialogues[d]['turns'][t] = turn

            with open(out_files[o], 'w') as f:
                json.dump(test_dialogues, f)

    def run_epoch(self, epoch, mode):
        self.__counter = 0
        self.stats = {}

        filename = self.data_dir / "test.pkl"
        schema_filename = self.origin_dir / "test" / "schema.json"
        schema_vocab_filename = self.data_dir / "test_schema_vocab.pkl"
        schema_embed_filename = self.data_dir / "test_schema_embed.pkl"
        preds = {
            "act_preds": [],
            "req_preds": [],
            "slot_value_preds": []}

        schemas = json.load(open(schema_filename))
        schema_vocab = pickle.load(open(schema_vocab_filename, 'rb'))
        _, self.idx2service = schema_vocab[0]
        _, self.idx2intent = schema_vocab[1]
        _, self.idx2slot = schema_vocab[2]
        _, self.idx2act = schema_vocab[3]

        self.cat_slots = extract_cat_slots(schemas, schema_vocab)
        if self.sch_embed:
            self.model._net.load_sch_embed(schema_embed_filename, self.device)

        data_loader = create_data_loader(
            filename=filename,
            config=self.config,
            vocab=self.vocab,
            mode=mode)
        ebar = tqdm(
            data_loader,
            desc=f"[{mode.upper()}]",
            leave=False,
            position=1)

        for b, d in enumerate(ebar):
            if hasattr(self, "update_freq"):
                if (b + 1) % self.update_freq == 0 and mode == "train":
                    self.is_update = True
            d = transfer(d, self.device)
            self.__counter += d['n_data']
            output = self.model(d, testing=True)
            act_preds, req_preds, slot_value_preds = \
                self.get_prediction(d, output)
            preds['act_preds'] += act_preds
            preds['req_preds'] += req_preds
            preds['slot_value_preds'] += slot_value_preds

        ebar.close()
        return preds

    def get_prediction(self, batch, output):
        act_o, req_o, dec_o, cxt_o = output
        # active intent
        act_preds = [torch.argmax(o).item() for o in act_o]
        act_preds = [
            self.idx2intent[batch['active_intent']['intent_idx'][i][j]][1]
            for i, j in enumerate(act_preds)]
        # requested slots
        req_o = [torch.sigmoid(o).flatten() for o in req_o]
        req_preds = []
        for i, o in enumerate(req_o):
            pred = []
            for j, val in enumerate(o):
                if val >= 0.5:
                    pred.append(self.idx2slot[
                        batch['requested_slots']['slot_idx'][i][j]][1])
            req_preds.append(pred)
        # slot tagging
        cxt_preds = [torch.argmax(o, dim=1).tolist() for o in cxt_o]
        dec_preds = [torch.argmax(o, dim=2).tolist() for o in dec_o]
        ext_lists = batch['ext_list']
        # extract filling values
        slot_preds = []
        for cxt_pred, dec_pred, ext_list in zip(
                cxt_preds, dec_preds, ext_lists):
            slot_preds.append(
                extract_values(
                    dec_pred, cxt_pred,
                    self.dc_gate, self.multi_value,
                    self.vocab, ext_list))

        final_slot_preds = []
        # convert categircal slot to possible values
        for didx, (preds, is_cateogircal, possible_values) in enumerate(
                zip(
                    slot_preds,
                    batch['slot_filling']['is_categorical'],
                    batch['slot_filling']['possible_values'])):
            final_preds = [[] for _ in preds]
            for sidx, (pred, flag, values) in enumerate(
                    zip(preds, is_cateogircal, possible_values)):
                if len(pred) == 0:
                    continue
                for pidx, p in enumerate(pred):
                    if len(p) == 0:
                        continue
                    if flag:
                        try:
                            words = self.tokenizer(p)
                            embs = [
                                self.emb[word.text] for word in words
                                if word in self.emb]
                            embs = np.mean(embs, axis=0)
                            val_emb = []
                            for v in values:
                                val_emb.append(
                                    np.mean(
                                        [
                                            self.emb[word.text]
                                            for word in
                                            self.tokenizer(v)
                                            if word in self.emb],
                                        axis=0))
                            final_preds[sidx].append(values[
                                self.get_most_likely(
                                    embs, val_emb, self.similarity)])
                        except IndexError:
                            pass
                    elif self.fix_syntax:
                        for mark in self.trim_front:
                            try:
                                if mark in p and p[p.index(mark) - 1] == " ":
                                    idx = p.index(mark)
                                    p = p[:idx - 1] + p[idx:]
                            except IndexError:
                                pass
                        for mark in self.trim_back:
                            try:
                                if mark in p and p[p.index(mark) + 1] == " ":
                                    idx = p.index(mark)
                                    p = p[:idx - 1] + p[idx:]
                            except IndexError:
                                pass
                        final_preds[sidx].append(p)
            final_slot_preds.append(final_preds)

        slot_value_preds = []
        for i, values in enumerate(final_slot_preds):
            preds = {}
            for j, vals in enumerate(values):
                if len(vals) > 0:
                    slot_idx = batch['requested_slots']['slot_idx'][i][j]
                    slot = self.idx2slot[slot_idx][1]
                    preds[slot] = [val for val in vals if len(val) > 0]
            slot_value_preds.append(preds)

        return act_preds, req_preds, slot_value_preds

    def get_most_likely(self, emb, candidates, metric='cos'):
        if metric == 'l2':
            likelihood = [LA.norm(emb, b) for b in candidates]
            return np.argmin(likelihood, axis=0)
        elif metric == 'cos':
            likelihood = [
                np.dot(emb, b) / (LA.norm(emb) * LA.norm(b))
                for b in candidates]
            return np.argmax(likelihood, axis=0)
