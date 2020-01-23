import argparse
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from box import Box
from pathlib import Path

from modules.logger import create_logger
from modules.utils import create_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of config file')
    args = parser.parse_args()
    return vars(args)


def get_matrix(tensor):
    embed = tensor.numpy()
    lengths = np.sqrt((embed ** 2).sum(axis=1)).reshape(1, -1)
    matrix = np.matmul(embed, embed.T)
    matrix = matrix / lengths / lengths.T
    return matrix


def plot(matrix, labels, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(
        ax.get_xticklabels(), rotation=45,
        ha="right", rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig(filename)


def main(config_path):
    config = Box.from_yaml(config_path.open())
    logger = create_logger(name="MAIN")
    logger.info(f'[-] Config loaded from {config_path}')

    data_dir = Path(config.data.data_dir)
    save_dir = Path(config.data.save_dir)
    fig_dir = save_dir / "fig"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    train_vocab_file = save_dir / "train_schema_vocab.pkl"
    valid_vocab_file = save_dir / "valid_schema_vocab.pkl"
    train_embed_file = save_dir / "train_schema_embed.pkl"
    valid_embed_file = save_dir / "valid_schema_embed.pkl"

    train_vocab = pickle.load(open(train_vocab_file, 'rb'))
    valid_vocab = pickle.load(open(valid_vocab_file, 'rb'))
    train_embed = pickle.load(open(train_embed_file, 'rb'))
    valid_embed = pickle.load(open(valid_embed_file, 'rb'))

    plt.rcParams.update({'font.size': 8})

    _, idx2service = train_vocab[0]
    matrix = get_matrix(train_embed[0])
    plot(matrix, idx2service, fig_dir / "train_service.pdf")

    _, idx2service = valid_vocab[0]
    matrix = get_matrix(valid_embed[0])
    plot(matrix, idx2service, fig_dir / "valid_service.pdf")

    plt.rcParams.update({'font.size': 6})

    _, idx2intent = train_vocab[1]
    matrix = get_matrix(train_embed[1])
    idx2intent = [intent for service, intent in idx2intent]
    plot(matrix, idx2intent, fig_dir / "train_intent.pdf")

    _, idx2intent = valid_vocab[1]
    matrix = get_matrix(valid_embed[1])
    idx2intent = [intent for service, intent in idx2intent]
    plot(matrix, idx2intent, fig_dir / "valid_intent.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
