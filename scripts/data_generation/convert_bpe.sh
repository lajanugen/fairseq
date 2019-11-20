DATA_DIR=/tmp/data
OUTPUT_DIR=/tmp/data_bpe
GPT_DIR=/tmp/gpt2_bpe

mkdir -p $OUTPUT_DIR

mkdir -p $GPT_DIR
wget -O $GPT_DIR/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O $GPT_DIR/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
wget -O $GPT_DIR/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

for TASK in train test; do
    for SPLIT in train valid; do
        python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json $GPT_DIR/encoder.json \
            --vocab-bpe $GPT_DIR/vocab.bpe \
            --inputs $DATA_DIR/$TASK.$SPLIT \
            --outputs $OUTPUT_DIR/$TASK.$SPLIT \
            --keep-empty \
            --workers 40
    done
done
