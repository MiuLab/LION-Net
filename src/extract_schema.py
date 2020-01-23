import argparse
import ipdb
import json
import numpy as np
import pickle
import torch
import spacy
import sys

from box import Box
from pathlib import Path
from tqdm import tqdm

from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertConfig, BertModel

from modules.logger import create_logger
from modules.utils import create_device

nlp = spacy.load('en')
spacy_tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of config file')
    args = parser.parse_args()
    return vars(args)


def get_rep(sent, model, tokenizer, layer, pooling, device):
    ids = [CLS] + tokenizer.encode(sent) + [SEP]
    input_ids = torch.LongTensor(ids).reshape(1, -1)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        _, _, hiddens = model(input_ids)
        hidden = hiddens[layer]
        if pooling == "mean":
            hidden = hidden.mean(dim=1).squeeze(0)
        elif pooling == "max":
            hidden = hidden.max(dim=1)[0].squeeze(0)

    return hidden.cpu()


def extract_words(string):
    words = []
    for c in string:
        if c == c.upper():
            words.append("")
        words[-1] += c
    return words


def extract(schema_file, concat_name=False):
    schemas = json.load(open(schema_file, 'r'))
    service2idx, idx2service = {}, []
    intent2idx, idx2intent = {}, []
    slot2idx, idx2slot = {}, []
    act2idx, idx2act = {}, []
    service2cat = {}
    slot2values = {}

    service_desc, intent_desc, slot_desc = [], [], []

    for schema in schemas:
        service_name = schema['service_name']
        service2idx[service_name] = len(idx2service)
        service2cat[service_name] = []
        idx2service.append(service_name)
        if concat_name:
            _service_name = service_name.replace("_", " ")
            service_desc.append(
                _service_name + '. ' + schema['description'])
        else:
            service_desc.append(schema['description'])

        for intent in schema['intents']:
            intent_name = intent['name']
            intent2idx[(service_name, intent_name)] = len(idx2intent)
            idx2intent.append((service_name, intent_name))
            if concat_name:
                _intent_name = ' '.join(extract_words(intent_name))
                intent_desc.append(
                    _intent_name + '. ' + intent['description'])
            else:
                intent_desc.append(intent['description'])
        intent2idx[(service_name, "NONE")] = len(idx2intent)
        idx2intent.append((service_name, "NONE"))
        intent_desc.append("")

        for slot in schema['slots']:
            slot_name = slot['name']
            slot2idx[(service_name, slot_name)] = len(idx2slot)
            idx2slot.append((service_name, slot_name))
            if concat_name:
                _slot_name = slot_name.replace("_", " ")
                slot_desc.append(_slot_name + '. ' + slot['description'])
            else:
                slot_desc.append(slot['description'])
            service2cat[service_name].append(slot['is_categorical'])
            slot2values[len(idx2slot) - 1] = slot['possible_values']

    idx2act = [
        "INFORM", "REQUEST", "CONFIRM", "OFFER",
        "NOTIFY_SUCCESS", "NOTIFY_FAILURE",
        "INFORM_COUNT", "OFFER_INTENT", "REQ_MORE", "GOODBYE"]
    act2idx = {act: idx for idx, act in enumerate(idx2act)}

    schema_vocab = [
        [service2idx, idx2service],
        [intent2idx, idx2intent],
        [slot2idx, idx2slot],
        [act2idx, idx2act],
        service2cat,
        slot2values]
    desc = [service_desc, intent_desc, slot_desc]
    return schema_vocab, desc


def main(config_path):
    config = Box.from_yaml(config_path.open())
    torch.cuda.set_device(config.train.device)
    logger = create_logger(name="MAIN")
    logger.info(f'[-] Config loaded from {config_path}')

    data_dir = Path(config.data.data_dir)
    save_dir = Path(config.data.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    transfo_dir = Path(config.data.transfo_dir)
    device = create_device(config.train.device)

    tokenizer = BertTokenizer.from_pretrained(
        transfo_dir,
        do_lower_case=(not config.data.cased))

    global CLS
    global SEP
    global PAD
    CLS, SEP, PAD = tokenizer.convert_tokens_to_ids(
        ["[CLS]", "[SEP]", "[PAD]"])

    bert_config = BertConfig.from_pretrained(transfo_dir)
    # To extract representations from other layers
    bert_config.output_hidden_states = True
    model = BertModel(bert_config)
    model.to(device)
    model.eval()

    train_file = data_dir / "train" / "schema.json"
    train_vocab_file = save_dir / "train_schema_vocab.pkl"
    train_embed_file = save_dir / "train_schema_embed.pkl"
    train_desc_file = save_dir / "train_schema_desc.pkl"
    valid_file = data_dir / "dev" / "schema.json"
    valid_vocab_file = save_dir / "valid_schema_vocab.pkl"
    valid_embed_file = save_dir / "valid_schema_embed.pkl"
    valid_desc_file = save_dir / "valid_schema_desc.pkl"
    if (data_dir / "test").exists():
        test_file = data_dir / "test" / "schema.json"
        test_vocab_file = save_dir / "test_schema_vocab.pkl"
        test_embed_file = save_dir / "test_schema_embed.pkl"
        test_desc_file = save_dir / "test_schema_desc.pkl"
    else:
        test_file = None
        test_vocab_file = None
        test_embed_file = None
        test_desc_file = None

    train_schema_vocab, train_desc = \
        extract(train_file, config.data.concat_name)
    valid_schema_vocab, valid_desc = \
        extract(valid_file, config.data.concat_name)
    if test_file is not None:
        test_schema_vocab, test_desc = \
            extract(test_file, config.data.concat_name)
    else:
        test_schema_vocab = test_desc = None

    pickle.dump(train_schema_vocab, open(train_vocab_file, 'wb'))
    pickle.dump(valid_schema_vocab, open(valid_vocab_file, 'wb'))
    if test_schema_vocab is not None:
        pickle.dump(test_schema_vocab, open(test_vocab_file, 'wb'))

    layer = config.data.schema.layer
    pooling = config.data.schema.pooling

    train_embed = []
    for desc in tqdm(train_desc, leave=False):
        embed = []
        for sent in tqdm(desc, leave=False):
            embed.append(
                get_rep(sent, model, tokenizer, layer, pooling, device))
        embed = torch.stack(embed)
        train_embed.append(embed)

    train_desc = [
        [
            [word.text.lower() for word in spacy_tokenizer(sent)]
            for sent in desc
        ]
        for desc in train_desc]

    pickle.dump(train_embed, open(train_embed_file, 'wb'))
    pickle.dump(train_desc, open(train_desc_file, 'wb'))

    valid_embed = []
    for desc in tqdm(valid_desc, leave=False):
        embed = []
        for sent in tqdm(desc, leave=False):
            embed.append(
                get_rep(sent, model, tokenizer, layer, pooling, device))
        embed = torch.stack(embed)
        valid_embed.append(embed)

    valid_desc = [
        [
            [word.text.lower() for word in spacy_tokenizer(sent)]
            for sent in desc
        ]
        for desc in valid_desc]

    pickle.dump(valid_embed, open(valid_embed_file, 'wb'))
    pickle.dump(valid_desc, open(valid_desc_file, 'wb'))

    if test_desc is None:
        exit()

    test_embed = []
    for desc in tqdm(test_desc, leave=False):
        embed = []
        for sent in tqdm(desc, leave=False):
            embed.append(
                get_rep(sent, model, tokenizer, layer, pooling, device))
        embed = torch.stack(embed)
        test_embed.append(embed)

    test_desc = [
        [
            [word.text.lower() for word in spacy_tokenizer(sent)]
            for sent in desc
        ]
        for desc in test_desc]

    pickle.dump(test_embed, open(test_embed_file, 'wb'))
    pickle.dump(test_desc, open(test_desc_file, 'wb'))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
