import torch
import torch.nn as nn


def extract_values(preds, gates, dc_gate, multi_value, vocab, ext_list):
    BOS = vocab.convert_tokens_to_indices(["<BOS>"])[0]
    EOS = vocab.convert_tokens_to_indices(["<EOS>"])[0]
    DONTCARE = vocab.convert_tokens_to_indices(["<DONTCARE>"])[0]
    values = []
    for pred, gate in zip(preds, gates):
        if dc_gate and gate == 2:
            values.append([[DONTCARE]])
        elif gate == 0:
            values.append([])
        else:
            if multi_value:
                values.append([])
                value = []
                for idx in pred:
                    if idx == EOS:
                        values[-1].append(value)
                        value = []
                    elif idx == BOS:
                        continue
                    else:
                        value.append(idx)
                if value != []:
                    values[-1].append(value)
                values[-1] = [value for value in values[-1] if value != []]
            else:
                value = []
                for idx in pred:
                    if idx == EOS:
                        break
                    elif idx == BOS:
                        continue
                    else:
                        value.append(idx)
                values.append([value])
    for i, vals in enumerate(values):
        for j, val in enumerate(vals):
            values[i][j] = ' '.join(
                vocab.convert_indices_to_tokens(val, ext_list))
    return values
