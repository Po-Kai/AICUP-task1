ckpt_id=10000
while [ $ckpt_id -le 100000 ]; do
    echo "ckpt_id: $ckpt_id"
    mkdir -p huggingface/v1/roberta_further_${ckpt_id}
    python convert_roberta_original_pytorch_checkpoint_to_pytorch.py \
        --roberta_checkpoint_path checkpoint/v1/roberta_further_${ckpt_id} \
        --pytorch_dump_folder_path huggingface/v1/roberta_further_${ckpt_id} \
        || exit
    
    let ckpt_id=ckpt_id+10000
done