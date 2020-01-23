#LION-Net: LIghtweight ONtology-independent Networks for Schema-Guided Dialogue State Generation
===
This repo contains source code of our DSTC8-trackIV 2020 paper "*LION-Net: LIghtweight ONtology-independent Networks for Schema-Guided Dialogue State Generation*"


## Requirements
* Python >= 3.6

Required python packages are listed in *requirements.txt*.

## Dataset

The dataset we used is Schema-Guided Dialogue State Tracking Dataset provided by Google.
> https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

## Preprocess

Download the dataset first and remember to download the GloVe word vectors.
> https://nlp.stanford.edu/projects/glove/

After downloading you need to put them into the directory you want.
#### Create dataset
    python3 preprocess.py
    python3 extract_schema.py

## Training
First, you need to make a copy of config.yaml.example and change the name to config.yaml
Then you can change the parameters in config.yaml and do the training.
Sample usage:

    python3 train.py

## Testing
Sample usage:

    python3 test.py


