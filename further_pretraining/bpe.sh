mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

for SPLIT in train valid; do \
    python multiprocessing_bpe_encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs datas/${SPLIT}.txt \
        --outputs datas/${SPLIT}.bpe \
        --keep-empty \
        --workers 8; \
done