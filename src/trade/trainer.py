import copy
import csv
import json
import numpy as np
import pickle
import re
import torch
import torch.nn as nn

from pathlib import Path
from time import gmtime, strftime
from tqdm import tqdm

from modules.dataset import create_data_loader, transfer
from modules.logger import create_logger
from modules.metrics import compute_active_intent_acc
from modules.metrics import compute_requested_slots_f1
from modules.metrics import compute_slot_filling_acc
from modules.utils import create_device, extract_cat_slots

from trade.model import Model
from trade.utils import extract_values


class Trainer:
    def __init__(self, config, device):
        for k, v in config.train.items():
            setattr(self, k, v)
        self.dc_gate = config.model_param.dc_gate
        self.sch_embed = (config.model_param.sch.type == "embed")

        self.logger = create_logger(name="TRAIN")
        self.origin_dir = Path(config.data.data_dir)
        self.data_dir = Path(config.data.save_dir)
        self.exp_dir = self.origin_dir / "exp" / config.model / self.exp

        self.config = config
        self.device = create_device(device)

        self.vocab = pickle.load(open(self.data_dir / "vocab.pkl", 'rb'))
        self.model = Model(
            config=config.model_param,
            vocab=self.vocab,
            device=self.device)
        self.__cur_epoch = 0

    def train(self):
        self.__checker()
        self.__initialize()
        for e in self.train_bar:
            self.model.train()
            self.stats = {}
            self.run_epoch(e, "train")
            train_stats = copy.deepcopy(self.stats)
            self.train_bar.write(self.__display(e + 1, "TRAIN", train_stats))
            self.model.eval()
            self.stats = {}
            self.run_epoch(e, "valid")
            valid_stats = copy.deepcopy(self.stats)
            display_stats = copy.deepcopy(self.stats)
            self.train_bar.write(self.__display(e + 1, "VALID", valid_stats))
            self.__logging(train_stats, valid_stats)
            self.model.save_state(e + 1, self.stats, self.exp_dir / "ckpt")
        self.train_bar.close()

    def run_epoch(self, epoch, mode):
        self.__counter = 0
        if self.show_metric is False and mode == "train":
            self.display_metric = False
        else:
            self.display_metric = True

        self.stats = {}
        self.stats["dec_loss"] = []
        self.stats["cxt_loss"] = []
        self.stats["req_loss"] = []
        self.stats["act_loss"] = []
        if self.display_metric:
            self.stats["goal_acc"] = []
            self.stats["joint_acc"] = []
            self.stats["req_f1"] = []
            self.stats["act_acc"] = []

        if mode == "train":
            filename = self.data_dir / "train.pkl"
            schema_filename = self.origin_dir / "train" / "schema.json"
            schema_vocab_filename = self.data_dir / "train_schema_vocab.pkl"
            schema_embed_filename = self.data_dir / "train_schema_embed.pkl"
        elif mode == "valid":
            filename = self.data_dir / "valid.pkl"
            schema_filename = self.origin_dir / "dev" / "schema.json"
            schema_vocab_filename = self.data_dir / "valid_schema_vocab.pkl"
            schema_embed_filename = self.data_dir / "valid_schema_embed.pkl"

        schemas = json.load(open(schema_filename))
        schema_vocab = pickle.load(open(schema_vocab_filename, 'rb'))
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
        self._sbar = tqdm(
            [0],
            desc=f"[Metric]",
            bar_format="{desc} {postfix}",
            leave=False,
            position=2)

        for b, d in enumerate(ebar):
            if hasattr(self, "update_freq"):
                if (b + 1) % self.update_freq == 0 and mode == "train":
                    self.is_update = True
            d = transfer(d, self.device)
            self.__counter += d['n_data']
            losses, metrics = self.run_batch(d, mode)
            self.stats["dec_loss"] += [l.item() for l in losses[0]]
            self.stats["cxt_loss"] += [l.item() for l in losses[1]]
            self.stats["req_loss"] += [l.item() for l in losses[2]]
            self.stats["act_loss"] += [l.item() for l in losses[3]]
            if self.display_metric:
                self.stats["goal_acc"] += metrics[0]
                self.stats["joint_acc"] += metrics[1]
                self.stats["req_f1"] += metrics[2]
                self.stats["act_acc"] += metrics[3]
            self._metric_display()

        ebar.close()
        self._sbar.close()

        for key in self.stats:
            self.stats[key] = np.mean(self.stats[key])

    def run_batch(self, batch, mode):
        output = self.model(batch, testing=(mode != "train"))
        losses = self.cal_loss(batch, output)
        dec_losses, cxt_losses, req_losses, act_losses = losses
        loss = (
            self.alpha * sum(dec_losses) / len(dec_losses) +
            self.beta * sum(cxt_losses) / len(cxt_losses) +
            self.gamma * sum(req_losses) / len(req_losses) +
            self.delta * sum(act_losses) / len(act_losses))

        if mode == "train":
            if hasattr(self, "update_freq"):
                loss /= self.update_freq
                loss.backward()
                if self.is_update:
                    if hasattr(self, "max_grad_norm"):
                        self.model.clip_grad(self.max_grad_norm)
                    self.model.update()
                    self.model.zero_grad()
                    self.is_update = False
            else:
                loss.backward()
                if hasattr(self, "max_grad_norm"):
                    self.model.clip_grad(self.max_grad_norm)
                self.model.update()
                self.model.zero_grad()

        losses = [
            dec_losses, cxt_losses,
            req_losses, act_losses]
        if self.display_metric:
            metrics = self.cal_metric(batch, output)
        else:
            metrics = []
        return losses, metrics

    def cal_loss(self, batch, output):
        act_o, req_o, dec_o, cxt_o = output
        act_l, req_l, dec_l, cxt_l = [], [], [], []
        for logit, label in zip(act_o, batch['active_intent']['label']):
            act_l.append(nn.CrossEntropyLoss()(logit, label))
        for logit, label in zip(req_o, batch['requested_slots']['label']):
            logit = logit.flatten()
            req_l.append(nn.BCEWithLogitsLoss()(logit, label))
        for idx, (cxt_logit, dec_logit) in enumerate(zip(cxt_o, dec_o)):
            cxt_label = batch['slot_filling']['context_label'][idx]
            cxt_l.append(nn.CrossEntropyLoss()(cxt_logit, cxt_label))
            val_label = batch['slot_filling']['value_ext_idx'][idx]
            val_mask = batch['slot_filling']['value_mask'][idx]
            nc = dec_logit.size(-1)
            probs = torch.gather(
                dec_logit.view(-1, nc), 1, val_label.view(-1, 1))
            dec_l.append(
                -torch.log(probs + 1e-8).masked_fill(
                    val_mask.view(-1, 1), 0).mean())
        return dec_l, cxt_l, req_l, act_l

    def cal_metric(self, batch, output):
        act_o, req_o, dec_o, cxt_o = output
        # active intent accuracy
        act_l = batch['active_intent']['label']
        act_acc = compute_active_intent_acc(act_o, act_l)
        # requested slots F1
        req_l = batch['requested_slots']['label']
        req_f1 = compute_requested_slots_f1(req_o, req_l)
        # slot tagging
        cxt_labels = [
            label.tolist() for label in
            batch['slot_filling']['context_label']]
        dec_labels = [
            label.tolist() for label in
            batch['slot_filling']['value_ext_idx']]
        cxt_preds = [torch.argmax(o, dim=1).tolist() for o in cxt_o]
        dec_preds = [torch.argmax(o, dim=2).tolist() for o in dec_o]
        ext_lists = batch['ext_list']
        # extract filling values
        slot_preds, slot_labels = [], []
        for cxt_pred, dec_pred, ext_list in zip(
                cxt_preds, dec_preds, ext_lists):
            slot_preds.append(
                extract_values(
                    dec_pred, cxt_pred,
                    self.dc_gate, self.multi_value,
                    self.vocab, ext_list))
        for cxt_label, dec_label, ext_list in zip(
                cxt_labels, dec_labels, ext_lists):
            slot_labels.append(
                extract_values(
                    dec_label, cxt_label,
                    self.dc_gate, self.multi_value,
                    self.vocab, ext_list))
        slot_idxes = [
            indices.tolist() for indices in
            batch['slot_filling']['value_slot_idx']]
        cat_tags = [
            [idx in self.cat_slots for idx in indices]
            for indices in slot_idxes]
        # calculate accuracy
        goal_accs, joint_accs = [], []
        for slot_pred, slot_label, cat_tag in zip(
                slot_preds, slot_labels, cat_tags):
            active_flags, value_accs = \
                compute_slot_filling_acc(slot_pred, slot_label, cat_tag)
            if len(value_accs) == 0:
                continue
            joint_accs.append(np.prod(value_accs))
            active_accs = [
                acc for acc, flag in zip(value_accs, active_flags)
                if flag is True]
            if active_accs != []:
                goal_accs.append(np.mean(active_accs))

        return goal_accs, joint_accs, req_f1, act_acc

    def __checker(self):
        ckpt_dir = self.exp_dir / "ckpt"
        if hasattr(self, "load_path"):
            self.logger.info(f"[*] Start training from {self.load_path}")
            self.model.load_state(
                self.load_path,
                getattr(self, "load_optim", True))
            if not ckpt_dir.is_dir():
                ckpt_dir.mkdir(parents=True)
        elif ckpt_dir.is_dir():
            files = list(ckpt_dir.glob("epoch*"))
            if files != []:
                files.sort()
                self.__cur_epoch = \
                    int(re.search('\d+', Path(files[-1]).name)[0])
            if self.__cur_epoch > 0:
                if self.__cur_epoch < self.epochs:
                    self.logger.info(
                        f"[*] Resume training (epoch {self.__cur_epoch+1}).")
                    self.model.load_state(
                        files[-1],
                        getattr(self, "load_optim", True))
                else:
                    while True:
                        retrain = input((
                            "The experiment is complete. "
                            "Do you want to re-train the model? y/[N] "))
                        if retrain in ['y', 'Y']:
                            self.__cur_epoch = 0
                            break
                        elif retrain in ['n', 'N', '']:
                            self.logger.info("[*] Quit the process...")
                            exit()
                    self.logger.info("[*] Start the experiment.")
        else:
            ckpt_dir.mkdir(parents=True)

    def __display(self, epoch, mode, stats):
        blank = "----"
        string = f"{epoch:>10} {mode:>10} "
        keys = [
            "dec_loss", "cxt_loss", "req_loss", "act_loss",
            "goal_acc", "joint_acc", "req_f1", "act_acc"]
        for key in keys:
            if key in stats:
                string += f"{np.mean(stats[key]):>10.2f}"
            else:
                string += f"{blank:>10}"
        return f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] " + string

    def _metric_display(self):
        postfix = '\b\b'
        losses, metrics = [], []
        losses.append(["d_loss", f"{np.mean(self.stats['dec_loss']):5.2f}"])
        losses.append(["c_loss", f"{np.mean(self.stats['cxt_loss']):5.2f}"])
        if self.display_metric:
            metrics.append(
                ["g_acc", f"{np.mean(self.stats['goal_acc']):5.2f}"])
            metrics.append(
                ["j_acc", f"{np.mean(self.stats['joint_acc']):5.2f}"])
        if self.gamma != 0:
            losses.append(
                ["r_loss", f"{np.mean(self.stats['req_loss']):5.2f}"])
            if self.display_metric:
                metrics.append(
                    ["r_f1", f"{np.mean(self.stats['req_f1']):5.2f}"])
        if self.delta != 0:
            losses.append(
                ["a_loss", f"{np.mean(self.stats['act_loss']):5.2f}"])
            if self.display_metric:
                metrics.append(
                    ["a_acc", f"{np.mean(self.stats['act_acc']):5.2f}"])

        postfix += ', '.join(f"{m}: {v}" for m, v in losses + metrics)
        self._sbar.set_postfix_str(postfix)

    def __logging(self, train_stats, valid_stats):
        log = {}
        for key, value in train_stats.items():
            log[f"TRAIN_{key}"] = f"{value:.2f}"
        for key, value in valid_stats.items():
            log[f"VALID_{key}"] = f"{value:.2f}"
        self.__log_writer.writerow(log)

    def __initialize(self):
        self.train_bar = tqdm(
            range(self.__cur_epoch, self.epochs),
            total=self.epochs,
            desc='[Total Progress]',
            initial=self.__cur_epoch,
            position=0)
        if hasattr(self, "update_freq"):
            self.is_update = False

        base_keys = ["EPOCH", "MODE"]
        loss_keys = ["D_LOSS", "C_LOSS"]
        if self.gamma != 0:
            loss_keys.append("R_LOSS")
        if self.delta != 0:
            loss_keys.append("A_LOSS")
        metric_keys = ["G_ACC", "J_ACC"]
        if self.gamma != 0:
            metric_keys.append("R_F1")
        if self.delta != 0:
            metric_keys.append("A_ACC")
        keys = base_keys + loss_keys + metric_keys
        string = ''.join(f"{key:>10}" for key in keys)
        string = f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] " + string
        self.train_bar.write(string)

        log_path = self.exp_dir / "log.csv"
        loss_keys = ["dec_loss", "cxt_loss"]
        if self.gamma != 0:
            loss_keys.append("req_loss")
        if self.delta != 0:
            loss_keys.append("act_loss")
        metric_keys = ["goal_acc", "joint_acc"]
        if self.gamma != 0:
            metric_keys.append("req_f1")
        if self.delta != 0:
            metric_keys.append("act_acc")
        train_fieldnames, valid_fieldnames = [], []
        for key in loss_keys:
            train_fieldnames.append(f'TRAIN_{key}')
            valid_fieldnames.append(f'VALID_{key}')
        for key in metric_keys:
            if self.show_metric:
                train_fieldnames.append(f'TRAIN_{key}')
            valid_fieldnames.append(f'VALID_{key}')
        fieldnames = train_fieldnames + valid_fieldnames

        if self.__cur_epoch == 0:
            self.__log_writer = csv.DictWriter(
                log_path.open(mode='w', buffering=1),
                fieldnames=fieldnames)
            self.__log_writer.writeheader()
        else:
            self.__log_writer = csv.DictWriter(
                log_path.open(mode='a', buffering=1),
                fieldnames=fieldnames)
