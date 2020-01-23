import numpy as np
import pickle
import torch

from functools import reduce
from operator import add
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class TorchDataset(Dataset):
    def __init__(self, data_path, max_context_len, full_slot, debug):
        self.data = []
        dataset = pickle.load(data_path.open('rb'))
        for example in dataset:
            context_len = len(example['utterances'][0])
            if context_len > max_context_len:
                index = context_len - max_context_len
                example['utterances'][0] = example['utterances'][0][index:]
            if not full_slot and len(example['slot_values']) == 0:
                continue
            self.data.append(example)
        if debug:
            self.data = self.data[:64]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def padding(seqs, max_seq_len=-1):
    if seqs == []:
        return []
    max_cur_len = max(len(seq) for seq in seqs)
    if max_seq_len == -1:
        max_len = max_cur_len
    else:
        max_len = max_seq_len
    seqs = [
        seq[:max_len] + [0 for _ in range(max_len - len(seq))]
        for seq in seqs]
    return seqs


def gen_collate_fn(config, vocab, mode="train"):

    EOS = vocab.convert_tokens_to_indices(["<EOS>"])[0]
    UNK = vocab.convert_tokens_to_indices(["<UNK>"])[0]
    PAD = vocab.convert_tokens_to_indices(["<PAD>"])[0]
    DONTCARE = vocab.convert_tokens_to_indices(["<DONTCARE>"])[0]
    vocab_size = len(vocab)

    if mode in ["train", "valid"]:
        mode_config = config.train
    elif mode == "test":
        mode_config = config.test

    if hasattr(mode_config, "full_slot"):
        full_slot = mode_config.full_slot or mode != "train"
    else:
        full_slot = (mode != "train")
    if hasattr(mode_config, "multi_value"):
        multi_value = mode_config.multi_value
    else:
        multi_value = False
    max_desc_len = mode_config.max_desc_len
    max_val_len = mode_config.max_val_len
    dc_gate = config.model_param.dc_gate
    save_dir = Path(config.data.save_dir)

    schema_vocab = pickle.load(
        open(save_dir / f"{mode}_schema_vocab.pkl", 'rb'))
    service2idx, idx2service = schema_vocab[0]
    intent2idx, idx2intent = schema_vocab[1]
    slot2idx, idx2slot = schema_vocab[2]

    slot2service = {}
    for idx, (service, slot) in enumerate(idx2slot):
        slot2service[idx] = service2idx[service]
    intent2service = {}
    for idx, (service, intent) in enumerate(idx2intent):
        intent2service[idx] = service2idx[service]

    if config.model_param.sch.type == "rnn":
        service2desc, intent2desc, slot2desc = pickle.load(
            open(save_dir / f"{mode}_schema_desc.pkl", 'rb'))

    def collate_fn(batch):
        n_data = len(batch)
        batch = {
            key: [data[key] for data in batch]
            for key in batch[0].keys()}
        output = {'n_data': n_data}
        context = [utter[0] for utter in batch['utterances']]
        context = [
            [idx if idx < vocab_size else UNK for idx in cxt]
            for cxt in context]
        output['context'] = [torch.LongTensor(cxt) for cxt in context]
        output['context'] = \
            [context.view(1, -1) for context in output['context']]
        output['ext_context'] = \
            [torch.LongTensor(utter[0]) for utter in batch['utterances']]
        output['ext_context'] = \
            [context.view(1, -1) for context in output['ext_context']]
        output['ext_list'] = [utter[1] for utter in batch['utterances']]
        # active intent
        output['active_intent'] = {}
        output['active_intent']['intent_idx'] = []
        output['active_intent']['service_idx'] = []
        if mode != "test":
            output['active_intent']['label'] = []
        if config.model_param.sch.type == "rnn":
            output['active_intent']['intent_desc'] = []
            output['active_intent']['service_desc'] = []
        for idx, (service, intent_list) in enumerate(zip(
                batch['service_idx'],
                batch['intent_list'])):
            intent_idx = intent_list
            service_idx = [service for _ in intent_idx]
            if mode != "test":
                label = batch['active_intent'][idx]
            if config.model_param.sch.type == "rnn":
                intent_desc = [intent2desc[idx] for idx in intent_idx]
                intent_desc = padding(intent_desc, max_desc_len)
                service_desc = [service2desc[idx] for idx in service_idx]
                service_desc = padding(service_desc, max_desc_len)
            output['active_intent']['intent_idx'].append(
                torch.LongTensor(intent_idx))
            output['active_intent']['service_idx'].append(
                torch.LongTensor(service_idx))
            if config.model_param.sch.type == "rnn":
                output['active_intent']['intent_desc'].append(
                    torch.LongTensor(intent_desc))
                output['active_intent']['service_desc'].append(
                    torch.LongTensor(service_desc))
            if mode != "test":
                output['active_intent']['label'].append(
                    torch.LongTensor([label]))
        # requested slots
        output['requested_slots'] = {}
        output['requested_slots']['slot_idx'] = []
        output['requested_slots']['service_idx'] = []
        if mode != "test":
            output['requested_slots']['label'] = []
        if config.model_param.sch.type == "rnn":
            output['requested_slots']['slot_desc'] = []
            output['requested_slots']['service_desc'] = []
        for idx, (service, slot_list) in enumerate(zip(
                batch['service_idx'],
                batch['slot_list'])):
            slot_idx = slot_list
            service_idx = [service for _ in slot_idx]
            if mode != "test":
                label = batch['requested_slots'][idx]
            if config.model_param.sch.type == "rnn":
                slot_desc = [slot2desc[idx] for idx in slot_idx]
                slot_desc = padding(slot_desc, max_desc_len)
                service_desc = [service2desc[idx] for idx in service_idx]
                service_desc = padding(service_desc, max_desc_len)
            output['requested_slots']['slot_idx'].append(
                torch.LongTensor(slot_idx))
            output['requested_slots']['service_idx'].append(
                torch.LongTensor(service_idx))
            if config.model_param.sch.type == "rnn":
                output['requested_slots']['slot_desc'].append(
                    torch.LongTensor(slot_desc))
                output['requested_slots']['service_desc'].append(
                    torch.LongTensor(service_desc))
            if mode != "test":
                output['requested_slots']['label'].append(
                    torch.FloatTensor(label))
        # slot filling
        output['slot_filling'] = {}
        output['slot_filling']['context_slot_idx'] = []
        output['slot_filling']['context_service_idx'] = []
        if mode != "test":
            output['slot_filling']['context_label'] = []
        output['slot_filling']['value_slot_idx'] = []
        output['slot_filling']['value_service_idx'] = []
        output['slot_filling']['is_categorical'] = []
        if mode != "test":
            output['slot_filling']['value_idx'] = []
            output['slot_filling']['value_ext_idx'] = []
            output['slot_filling']['value_mask'] = []
        if mode == "test":
            output['slot_filling']['possible_values'] = \
                batch['possible_values']
        output['slot_filling']['max_len'] = max_val_len
        if config.model_param.sch.type == "rnn":
            output['slot_filling']['context_slot_desc'] = []
            output['slot_filling']['context_service_desc'] = []
            output['slot_filling']['value_slot_desc'] = []
            output['slot_filling']['value_service_desc'] = []
        for index, (service, slot_list, is_cat) in enumerate(zip(
                batch['service_idx'],
                batch['slot_list'],
                batch['is_categorical'])):
            if mode != "test":
                slot_values = batch['slot_values'][index]
            if full_slot:
                slot_idx = slot_list
                service_idx = [service for _ in slot_idx]
                cat_flag = is_cat
                if mode != "test":
                    value_idx = [[EOS] for _ in slot_idx]
                    value_ext_idx = [[EOS] for _ in slot_idx]
                    context_label = [0 for _ in slot_idx]
                    for slot, values in slot_values:
                        pos = slot_idx.index(slot)
                        context_label[pos] = 1
                        if values[0] == [DONTCARE]:
                            if dc_gate:
                                context_label[pos] = 2
                            else:
                                value_idx[pos] = [DONTCARE, EOS]
                                value_ext_idx[pos] = [DONTCARE, EOS]
                            continue

                        if multi_value:
                            val = []
                            for value in values:
                                val += (value + [EOS])
                            value_idx[pos] = [
                                idx if idx < vocab_size else UNK
                                for idx in val]
                            value_ext_idx[pos] = val
                        else:
                            value_idx[pos] = [
                                idx if idx < vocab_size else UNK
                                for idx in values[0]]
                            value_idx[pos] = value_idx[pos] + [EOS]
                            value_ext_idx[pos] = values[0] + [EOS]

                if config.model_param.sch.type == "rnn":
                    slot_desc = [slot2desc[idx] for idx in slot_idx]
                    service_desc = [service2desc[idx] for idx in service_idx]
                    slot_desc = padding(slot_desc, max_desc_len)
                    service_desc = padding(service_desc, max_desc_len)

                context_slot_idx = torch.LongTensor(slot_idx)
                context_service_idx = torch.LongTensor(service_idx)
                value_slot_idx = torch.LongTensor(slot_idx)
                value_service_idx = torch.LongTensor(service_idx)
                is_categorical = torch.LongTensor(cat_flag)
                if mode != "test":
                    context_label = torch.LongTensor(context_label)
                    value_idx = torch.LongTensor(
                        padding(value_idx, max_val_len))
                    value_ext_idx = torch.LongTensor(
                        padding(value_ext_idx, max_val_len))
                    value_mask = (value_ext_idx == PAD)
                if config.model_param.sch.type == "rnn":
                    context_slot_desc = torch.LongTensor(slot_desc)
                    context_service_desc = torch.LongTensor(service_desc)
                    value_slot_desc = torch.LongTensor(slot_desc)
                    value_service_desc = torch.LongTensor(service_desc)
            else:
                context_slot_idx = slot_list
                context_service_idx = [service for _ in slot_list]
                context_label = [0 for _ in context_slot_idx]
                value_slot_idx = []
                value_service_idx = []
                cat_flag = []
                value_idx = []
                value_ext_idx = []
                for slot, values in slot_values:
                    value_slot_idx.append(slot)
                    value_service_idx.append(service)
                    pos = context_slot_idx.index(slot)
                    cat_flag.append(is_cat[pos])
                    context_label[pos] = 1
                    if values[0] == [DONTCARE]:
                        if dc_gate:
                            context_label[pos] = 2
                            value_idx.append([EOS])
                            value_ext_idx.append([EOS])
                        else:
                            value_idx.append([DONTCARE, EOS])
                            value_ext_idx.append([DONTCARE, EOS])
                        continue

                    if multi_value:
                        val = []
                        for value in values:
                            val += (value + [EOS])
                        value_idx.append([
                            idx if idx < vocab_size else UNK
                            for idx in val])
                        value_ext_idx.append(val)
                    else:
                        value_idx.append([
                            idx if idx < vocab_size else UNK
                            for idx in values[0]])
                        value_idx[-1] = value_idx[-1] + [EOS]
                        value_ext_idx.append(values[0] + [EOS])

                if config.model_param.sch.type == "rnn":
                    context_slot_desc = [
                        slot2desc[idx] for idx in context_slot_idx]
                    context_service_desc = [
                        service2desc[idx] for idx in context_service_idx]
                    value_slot_desc = [
                        slot2desc[idx] for idx in value_slot_idx]
                    value_service_desc = [
                        slot2desc[idx] for idx in value_service_idx]
                    context_slot_desc = \
                        padding(context_slot_desc, max_desc_len)
                    context_service_desc = \
                        padding(context_service_desc, max_desc_len)
                    value_slot_desc = \
                        padding(value_slot_desc, max_desc_len)
                    value_service_desc = \
                        padding(value_service_desc, max_desc_len)

                context_slot_idx = torch.LongTensor(context_slot_idx)
                context_service_idx = torch.LongTensor(context_service_idx)
                context_label = torch.LongTensor(context_label)
                value_slot_idx = torch.LongTensor(value_slot_idx)
                value_service_idx = torch.LongTensor(value_service_idx)
                is_categorical = torch.LongTensor(cat_flag)
                value_idx = torch.LongTensor(padding(value_idx, max_val_len))
                value_ext_idx = torch.LongTensor(
                    padding(value_ext_idx, max_val_len))
                value_mask = (value_ext_idx == PAD)
                if config.model_param.sch.type == "rnn":
                    context_slot_desc = torch.LongTensor(context_slot_desc)
                    context_service_desc = \
                        torch.LongTensor(context_service_desc)
                    value_slot_desc = torch.LongTensor(value_slot_desc)
                    value_service_desc = \
                        torch.LongTensor(value_service_desc)

            output['slot_filling']['context_slot_idx'].append(
                context_slot_idx)
            output['slot_filling']['context_service_idx'].append(
                context_service_idx)
            output['slot_filling']['value_slot_idx'].append(
                value_slot_idx)
            output['slot_filling']['value_service_idx'].append(
                value_service_idx)
            output['slot_filling']['is_categorical'].append(
                is_categorical)
            if mode != "test":
                output['slot_filling']['context_label'].append(
                    context_label)
                if mode == "train":
                    output['slot_filling']['value_idx'].append(value_idx)
                else:
                    output['slot_filling']['value_idx'].append(None)
                output['slot_filling']['value_ext_idx'].append(
                    value_ext_idx)
                output['slot_filling']['value_mask'].append(
                    value_mask)
            else:
                output['slot_filling']['value_idx'] = \
                    [None for _ in range(output['n_data'])]
            if config.model_param.sch.type == "rnn":
                output['slot_filling']['context_slot_desc'].append(
                    context_slot_desc)
                output['slot_filling']['context_service_desc'].append(
                    context_service_desc)
                output['slot_filling']['value_slot_desc'].append(
                    value_slot_desc)
                output['slot_filling']['value_service_desc'].append(
                    value_service_desc)

        return output

    return collate_fn


def create_data_loader(
        filename,
        config,
        vocab,
        mode="train"):
    collate_fn = gen_collate_fn(config, vocab, mode)
    if mode in ["train", "valid"]:
        max_context_len = config.train.max_context_len
        full_slot = config.train.full_slot if mode == "train" else True
        batch_size = config.train.batch_size
        debug = config.train.debug
    elif mode == "test":
        max_context_len = config.test.max_context_len
        full_slot = True
        batch_size = config.test.batch_size
        debug = config.test.debug
    dataset = TorchDataset(Path(filename), max_context_len, full_slot, debug)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn)
    return data_loader


def transfer(batch, device):
    batch['context'] = [
        context.to(device=device)
        for context in batch['context']]
    batch['ext_context'] = [
        context.to(device=device)
        for context in batch['ext_context']]
    batch['active_intent']['intent_idx'] = [
        intent_idx.to(device=device)
        for intent_idx in batch['active_intent']['intent_idx']]
    batch['active_intent']['service_idx'] = [
        service_idx.to(device=device)
        for service_idx in batch['active_intent']['service_idx']]
    if 'label' in batch['active_intent']:
        batch['active_intent']['label'] = [
            label.to(device=device)
            for label in batch['active_intent']['label']]
    if 'intent_desc' in batch['active_intent']:
        batch['active_intent']['intent_desc'] = [
            intent_desc.to(device=device)
            for intent_desc in batch['active_intent']['intent_desc']]
    if 'service_desc' in batch['active_intent']:
        batch['active_intent']['service_desc'] = [
            service_desc.to(device=device)
            for service_desc in batch['active_intent']['service_desc']]
    batch['requested_slots']['slot_idx'] = [
        slot_idx.to(device=device)
        for slot_idx in batch['requested_slots']['slot_idx']]
    batch['requested_slots']['service_idx'] = [
        service_idx.to(device=device)
        for service_idx in batch['requested_slots']['service_idx']]
    if 'label' in batch['requested_slots']:
        batch['requested_slots']['label'] = [
            label.to(device=device)
            for label in batch['requested_slots']['label']]
    if 'slot_desc' in batch['requested_slots']:
        batch['requested_slots']['slot_desc'] = [
            slot_desc.to(device=device)
            for slot_desc in batch['requested_slots']['slot_desc']]
    if 'service_desc' in batch['requested_slots']:
        batch['requested_slots']['service_desc'] = [
            service_desc.to(device=device)
            for service_desc in batch['requested_slots']['service_desc']]
    batch['slot_filling']['context_slot_idx'] = [
        slot_idx.to(device=device)
        for slot_idx in batch['slot_filling']['context_slot_idx']]
    batch['slot_filling']['context_service_idx'] = [
        service_idx.to(device=device)
        for service_idx in batch['slot_filling']['context_service_idx']]
    if 'context_label' in batch['slot_filling']:
        batch['slot_filling']['context_label'] = [
            label.to(device=device)
            for label in batch['slot_filling']['context_label']]
    batch['slot_filling']['value_slot_idx'] = [
        slot_idx.to(device=device)
        for slot_idx in batch['slot_filling']['value_slot_idx']]
    batch['slot_filling']['value_service_idx'] = [
        service_idx.to(device=device)
        for service_idx in batch['slot_filling']['value_service_idx']]
    batch['slot_filling']['value_idx'] = [
        value_idx.to(device=device)
        if value_idx is not None else None
        for value_idx in batch['slot_filling']['value_idx']]
    if 'value_ext_idx' in batch['slot_filling']:
        batch['slot_filling']['value_ext_idx'] = [
            value_idx.to(device=device)
            for value_idx in batch['slot_filling']['value_ext_idx']]
    if 'value_mask' in batch['slot_filling']:
        batch['slot_filling']['value_mask'] = [
            mask.to(device=device)
            for mask in batch['slot_filling']['value_mask']]
    if 'context_slot_desc' in batch['slot_filling']:
        batch['slot_filling']['context_slot_desc'] = [
            slot_desc.to(device=device)
            for slot_desc in batch['slot_filling']['context_slot_desc']]
    if 'context_service_desc' in batch['slot_filling']:
        batch['slot_filling']['context_service_desc'] = [
            service_desc.to(device=device)
            for service_desc in
            batch['slot_filling']['context_service_desc']]
    if 'value_slot_desc' in batch['slot_filling']:
        batch['slot_filling']['value_slot_desc'] = [
            slot_desc.to(device=device)
            for slot_desc in batch['slot_filling']['value_slot_desc']]
    if 'value_service_desc' in batch['slot_filling']:
        batch['slot_filling']['value_service_desc'] = [
            service_desc.to(device=device)
            for service_desc in
            batch['slot_filling']['value_service_desc']]
    return batch

