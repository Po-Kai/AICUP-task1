wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref datas/train.bpe \
    --validpref datas/valid.bpe \
    --destdir data-bin/ \
    --workers 8