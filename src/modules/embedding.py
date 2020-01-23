import glob
import pickle
import torch
import torch.nn as nn

from tqdm import tqdm

from modules.logger import create_logger
from modules.utils import get_num_lines


class Embedding(nn.Module):
    def __init__(self, config, vocab):
        super(Embedding, self).__init__()
        logger = create_logger(name="EMBED")
        UNK = vocab.convert_tokens_to_indices(["<UNK>"])[0]
        PAD = vocab.convert_tokens_to_indices(["<PAD>"])[0]
        if hasattr(config.emb, 'embed_path'):
            weight = pickle.load(open(config.emb.embed_path, 'rb'))
            self.model = nn.Embedding.from_pretrained(
                weight,
                freeze=config.emb.freeze,
                padding_idx=PAD)
        else:
            self.model = nn.Embedding(
                    len(vocab),
                    config.emb.dim,
                    padding_idx=PAD)
            logger.info("[-] Train from scratch.")

    def forward(self, i):
        return self.model(i)

    @property
    def num_embeddings(self):
        return self.model.num_embeddings

    @property
    def embedding_dim(self):
        return self.model.embedding_dim
