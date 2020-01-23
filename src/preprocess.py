import argparse
import copy
import ipdb
import json
import numpy as np
import pickle
import spacy
import sys
import torch

from box import Box
from collections import defaultdict, Counter
from functools import reduce
from operator import add
from pathlib import Path
from tqdm import tqdm

from modules.logger import create_logger
from modules.utils import get_num_lines
from modules.vocab import Vocab

nlp = spacy.load('en')
tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of config file')
    args = parser.parse_args()
    return vars(args)


def gen_lists(services, fields):
    lists = [[] for _ in services]
    for i, service in enumerate(services):
        for j, field in enumerate(fields):
            if service == field[0]:
                lists[i].append(j)
    return lists


def count_words(dialogues, schemas):
    counter = Counter()
    for dialog in tqdm(dialogues, leave=False):
        for turn in dialog['turns']:
            utterance = tokenizer(turn['utterance'])
            counter.update([word.text for word in utterance])
    for schema in tqdm(schemas, leave=False):
        description = tokenizer(schema['description'])
        counter.update([word.text.lower() for word in description] * 50)
        for intent in schema['intents']:
            description = tokenizer(intent['description'])
            counter.update([word.text.lower() for word in description] * 50)
        for slot in schema['slots']:
            description = tokenizer(slot['description'])
            counter.update([word.text.lower() for word in description] * 50)
            if slot['is_categorical']:
                for value in slot['possible_values']:
                    value = tokenizer(value)
                    counter.update([word.text for word in value] * 50)

    return counter


def build_dataset(dialogues, vocab, schema_vocab, schemas):

    service2idx, idx2service = schema_vocab[0]
    intent2idx, idx2intent = schema_vocab[1]
    slot2idx, idx2slot = schema_vocab[2]
    act2idx, idx2act = schema_vocab[3]
    service2cat = schema_vocab[4]
    slot2values = schema_vocab[5]

    intent_lists = gen_lists(idx2service, idx2intent)
    slot_lists = gen_lists(idx2service, idx2slot)

    num_service = len(idx2service)
    num_intent = len(idx2intent)
    num_slot = len(idx2slot)

    DONTCARE = vocab.convert_tokens_to_indices(["<DONTCARE>"])[0]

    dataset = []
    for dialog in tqdm(dialogues):
        utterances = []
        slots = []
        for turn in dialog['turns']:
            utterance = [
                word.text for word in tokenizer(turn['utterance'])]
            utterance = ["<BOS>"] + utterance + ["<EOS>"]
            utterances += utterance

            if turn['speaker'] == "USER":
                intent_mask, slot_mask = [], []
                for frame in turn['frames']:
                    example = {}
                    example['utterances'] = copy.deepcopy(utterances)
                    example['utterances'] = list(
                        vocab.convert_tokens_to_indices(
                            example['utterances'], mode='ext'))
                    # extract intents and slots
                    service_name = frame['service']
                    service_idx = service2idx[service_name]
                    example['service_idx'] = service_idx
                    example['intent_list'] = intent_lists[service_idx]
                    example['slot_list'] = slot_lists[service_idx]
                    example['is_categorical'] = service2cat[service_name]
                    example['possible_values'] = []
                    for slot_idx in example['slot_list']:
                        example['possible_values'].append(
                            slot2values[slot_idx])

                    if 'state' in frame:
                        # get requested slots
                        requested_slots = np.zeros(
                            (len(example['slot_list'])))
                        for slot_name in frame['state']['requested_slots']:
                            slot_idx = slot2idx[(service_name, slot_name)]
                            pos = example['slot_list'].index(slot_idx)
                            requested_slots[pos] = 1
                        example['requested_slots'] = requested_slots
                        # get active intent
                        intent_name = frame['state']['active_intent']
                        intent_idx = intent2idx[(service_name, intent_name)]
                        active_intent = \
                            example['intent_list'].index(intent_idx)
                        example['active_intent'] = active_intent
                        # get slots and values
                        example['slot_values'] = []
                        ext_list = example['utterances'][1]

                        slot_values = frame['state']['slot_values']
                        for slot_name, values in slot_values.items():
                            slot_idx = slot2idx[(service_name, slot_name)]
                            value_idxes = []
                            for value in values:
                                if value == "dontcare":
                                    value_idxes.append([DONTCARE])
                                else:
                                    value = [
                                        word.text
                                        for word in tokenizer(value)]
                                    value_idxes.append(
                                        vocab.convert_tokens_to_indices(
                                            value,
                                            mode='ext',
                                            ext_list=ext_list))
                            example['slot_values'].append(
                                [slot_idx, value_idxes])
                    dataset.append(example)
    return dataset


def main(config_path):
    logger = create_logger(name="DATA")
    config = Box.from_yaml(config_path.open())
    data_dir = Path(config.data.data_dir)
    save_dir = Path(config.data.save_dir)

    sd_train_files = [
        data_dir / "train" / f"dialogues_{idx:0>3}.json"
        for idx in range(1, 44)]
    md_train_files = [
        data_dir / "train" / f"dialogues_{idx:0>3}.json"
        for idx in range(44, 128)]

    sd_valid_files = [
        data_dir / "dev" / f"dialogues_{idx:0>3}.json"
        for idx in range(1, 8)]
    md_valid_files = [
        data_dir / "dev" / f"dialogues_{idx:0>3}.json"
        for idx in range(8, 21)]

    is_test = (data_dir / "test").exists()
    # Wait for test data release
    if is_test:
        sd_test_files = [
            data_dir / "test" / f"dialogues_{idx:0>3}.json"
            for idx in range(1, 8)]
        md_test_files = [
            data_dir / "test" / f"dialogues_{idx:0>3}.json"
            for idx in range(8, 21)]

    train_files, valid_files, test_files = [], [], []

    use_sd = config.data.train.single_domain
    use_md = config.data.train.multi_domain
    assert use_sd or use_md, "Please use at least one part of dataset"
    if use_sd:
        train_files += sd_train_files
    if use_md:
        train_files += md_train_files

    use_sd = config.data.valid.single_domain
    use_md = config.data.valid.multi_domain
    assert use_sd or use_md, "Please use at least one part of dataset"
    if use_sd:
        valid_files += sd_valid_files
    if use_md:
        valid_files += md_valid_files

    if is_test:
        use_sd = config.data.test.single_domain
        use_md = config.data.test.multi_domain
        assert use_sd or use_md, "Please use at least one part of dataset"
        if use_sd:
            test_files += sd_test_files
        if use_md:
            test_files += md_test_files

    train_schema_file = data_dir / "train" / "schema.json"
    valid_schema_file = data_dir / "dev" / "schema.json"
    train_schemas = json.load(open(train_schema_file))
    valid_schemas = json.load(open(valid_schema_file))

    train_schema_vocab_file = save_dir / "train_schema_vocab.pkl"
    valid_schema_vocab_file = save_dir / "valid_schema_vocab.pkl"
    train_schema_vocab = pickle.load(open(train_schema_vocab_file, 'rb'))
    valid_schema_vocab = pickle.load(open(valid_schema_vocab_file, 'rb'))

    train_dialogues = []
    for f in train_files:
        train_dialogues.extend(json.load(open(f)))

    valid_dialogues = []
    for f in valid_files:
        valid_dialogues.extend(json.load(open(f)))

    if is_test:
        test_schema_file = data_dir / "test" / "schema.json"
        test_schemas = json.load(open(test_schema_file))
        test_schema_vocab_file = save_dir / "test_schema_vocab.pkl"
        test_schema_vocab = pickle.load(open(test_schema_vocab_file, 'rb'))

        test_dialogues = []
        for f in test_files:
            test_dialogues.extend(json.load(open(f)))

    # Build vocab
    counter = count_words(train_dialogues, train_schemas)
    vocab_size = config.data.vocab_size
    vocab = Vocab(counter, vocab_size)
    vocab_path = save_dir / "vocab.pkl"
    logger.info(f"[-] Vocab size: {vocab_size}")
    logger.info(f"[-] Full vocab size: {len(vocab._idx2token)}")
    logger.info(f"[*] Dump vocab to {vocab_path}")
    pickle.dump(vocab, open(vocab_path, 'wb'))

    # Generate embeddings
    UNK = vocab.convert_tokens_to_indices(["<UNK>"])[0]
    PAD = vocab.convert_tokens_to_indices(["<PAD>"])[0]
    with open(config.data.embed_path, 'r') as file:
        line = next(file)
        emb_dim = len(line.strip().split()) - 1
    cover = 0
    weight = torch.zeros(len(vocab), emb_dim)
    with open(config.data.embed_path, 'r') as file:
        for line in tqdm(
                file,
                total=get_num_lines(config.data.embed_path),
                leave=False):
            data = line.strip().split(' ')
            token, emb = data[0], list(map(float, data[1:]))
            idx = vocab.convert_tokens_to_indices([token])[0]
            if len(emb) == emb_dim and idx != UNK:
                cover += 1
                weight[idx] = torch.FloatTensor(emb)
    weight[UNK] = 0.0
    weight[PAD] = 0.0
    logger.info((
        f"[-] Coverage: {cover}/{len(vocab)} "
        f"({cover / len(vocab) * 100:.2f}%)."))
    pickle.dump(weight, open(config.model_param.emb.embed_path, 'wb'))

    # Build dataset
    dataset = build_dataset(
        train_dialogues, vocab, train_schema_vocab, train_schemas)
    logger.info(f"[-] {len(dataset)} Examples for training")
    pickle.dump(dataset, open(save_dir / "train.pkl", 'wb'))
    dataset = build_dataset(
        valid_dialogues, vocab, valid_schema_vocab, valid_schemas)
    logger.info(f"[-] {len(dataset)} Examples for validating")
    pickle.dump(dataset, open(save_dir / "valid.pkl", 'wb'))
    if is_test:
        dataset = build_dataset(
            test_dialogues, vocab, test_schema_vocab, test_schemas)
        logger.info(f"[-] {len(dataset)} Examples for testing")
        pickle.dump(dataset, open(save_dir / "test.pkl", 'wb'))

    # Convert schema desc
    schema_desc = pickle.load(open(save_dir / "train_schema_desc.pkl", 'rb'))
    schema_desc = [
        [vocab.convert_tokens_to_indices(sent) for sent in desc]
        for desc in schema_desc]
    pickle.dump(schema_desc, open(save_dir / "train_schema_desc.pkl", 'wb'))
    schema_desc = pickle.load(open(save_dir / "valid_schema_desc.pkl", 'rb'))
    schema_desc = [
        [vocab.convert_tokens_to_indices(sent) for sent in desc]
        for desc in schema_desc]
    pickle.dump(schema_desc, open(save_dir / "valid_schema_desc.pkl", 'wb'))
    if is_test:
        schema_desc = pickle.load(
            open(save_dir / "test_schema_desc.pkl", 'rb'))
        schema_desc = [
            [vocab.convert_tokens_to_indices(sent) for sent in desc]
            for desc in schema_desc]
        pickle.dump(schema_desc, open(save_dir / "test_schema_desc.pkl", 'wb'))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
