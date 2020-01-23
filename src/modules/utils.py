import mmap
import torch


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def create_device(device):
    return \
        torch.device('cuda:{}'.format(device)) \
        if device >= 0 else torch.device('cpu')


def extract_cat_slots(schemas, schema_vocab):
    slot2idx = schema_vocab[2][0]
    cat_slots = set()
    for schema in schemas:
        service_name = schema['service_name']
        for slot in schema['slots']:
            slot_name = slot['name']
            if slot['is_categorical']:
                idx = slot2idx[(service_name, slot_name)]
                cat_slots.add(idx)
    return cat_slots
