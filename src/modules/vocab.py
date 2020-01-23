import glob
import json
import pickle

from pathlib import Path
from tqdm import tqdm


class Vocab:
    def __init__(self, counter, size):
        self._special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<DONTCARE>"]
        self._size = size
        self._idx2token = [token for token in self._special_tokens] + \
            [token for token, _ in counter.most_common()]
        self._token2idx = {
            token: idx for idx, token in enumerate(self._idx2token)}

    def convert_tokens_to_indices(self, tokens, mode="normal", ext_list=None):
        if '–' in tokens:
            for idx, token in enumerate(tokens):
                if '–' == token:
                    tokens[idx] = '-'
        indices = [
            self._token2idx[token]
            if token in self._token2idx else 1
            for token in tokens]
        if mode == "normal":
            indices = [idx if idx < self._size else 1 for idx in indices]
            return indices
        elif mode == "full":
            return indices
        elif mode == "ext":
            if ext_list is None:
                ext_list = []
                for pos, idx in enumerate(indices):
                    if idx >= self._size:
                        if idx in ext_list:
                            indices[pos] = self._size + ext_list.index(idx)
                        else:
                            indices[pos] = self._size + len(ext_list)
                            ext_list.append(idx)
                return indices, ext_list
            else:
                for pos, idx in enumerate(indices):
                    if idx >= self._size:
                        if idx in ext_list:
                            indices[pos] = self._size + ext_list.index(idx)
                        else:
                            indices[pos] = 1
                return indices

    def convert_indices_to_tokens(self, indices, ext_list=None):
        if ext_list is None:
            tokens = [self._idx2token[idx] for idx in indices]
        else:
            indices = [
                idx if idx < self._size else ext_list[idx - self._size]
                for idx in indices]
            tokens = [self._idx2token[idx] for idx in indices]
        return tokens

    def __len__(self):
        return self._size
