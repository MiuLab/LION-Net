import argparse
import ipdb
import json
import sys

from box import Box
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of config file')
    args = parser.parse_args()
    return vars(args)


def main(config_path):
    config = Box.from_yaml(config_path.open())
    data_dir = Path(config.data.data_dir)
    save_dir = Path(config.data.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    corpus_path = save_dir / "corpus.txt"

    files = list((data_dir / "train").glob("dialogues_*.json"))

    with open(corpus_path, 'w') as file:
        for f in tqdm(files, leave=False):
            dialogs = json.load(open(f))
            for dialog in tqdm(dialogs, leave=False):
                for turn in dialog['turns']:
                    file.write(f"{turn['utterance']}\n")
                file.write("\n")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
