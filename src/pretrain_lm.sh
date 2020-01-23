MODEL_DIR="/tmp/dstc8-track4/transfo/bert-base-cased/"
FINETUNED_DIR="/tmp/dstc8-track4/transfo/bert-base-cased-finetuned/"

EPOCHS=3
MAX_SEQ_LEN=64
ACC_STEPS=4

CONFIG_FILE=$1
SAVE_DIR=$(grep save_dir ${CONFIG_FILE} | cut -d: -f2 | tr -d '"')
CORPUS_FILE=$SAVE_DIR/corpus.txt

python3 lm_finetuning/extract_lm_corpus.py -c $CONFIG_FILE

python3 lm_finetuning/pregenerate_training_data.py \
--train_corpus $CORPUS_FILE \
--bert_model $MODEL_DIR \
--output_dir $SAVE_DIR/lm_finetuning \
--epochs_to_generate $EPOCHS \
--max_seq_len $MAX_SEQ_LEN

python3 lm_finetuning/finetune_on_pregenerated.py \
--pregenerated_data $SAVE_DIR/lm_finetuning \
--bert_model $MODEL_DIR \
--output_dir $FINETUNED_DIR \
--epochs $EPOCHS \
--gradient_accumulation_steps $ACC_STEPS \
--reduce_memory
