import torch
import torch.nn as nn

from collections import Counter
from fuzzywuzzy import fuzz


def compute_f1(ref, hyp):
    ref, hyp = Counter(ref), Counter(hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return f1, precision, recall


def compute_requested_slots_f1(outputs, labels):
    preds = [nn.Sigmoid()(o).flatten() for o in outputs]
    preds = [(pred >= 0.5).tolist() for pred in preds]
    preds = [
        [i for i, p in enumerate(pred) if p]
        for pred in preds]
    labels = [
        (label >= 0.5).flatten().tolist()
        for label in labels]
    labels = [
        [i for i, l in enumerate(label) if l]
        for label in labels]
    f1_scores = [
        compute_f1(label, pred)[0] for pred, label in
        zip(labels, preds)]
    return f1_scores


def compute_active_intent_acc(outputs, labels):
    preds = [torch.argmax(o).item() for o in outputs]
    labels = [label.item() for label in labels]
    act_acc = [
        pred == label for pred, label in
        zip(preds, labels)]
    return act_acc


def compute_slot_filling_acc(preds, labels, tags):
    active_flags, value_accs = [], []
    for pred, label, tag in zip(preds, labels, tags):
        if label != []:
            active_flags.append(True)
            if pred == []:
                value_accs.append(0.0)
            else:
                if tag:
                    acc = float(label[0] == pred[0])
                else:
                    acc = max([
                        fuzz.token_sort_ratio(l, pred[0]) / 100
                        for l in label])
                value_accs.append(acc)
        else:
            active_flags.append(False)
            if pred == []:
                value_accs.append(1.0)
            else:
                value_accs.append(0.0)
    return active_flags, value_accs
