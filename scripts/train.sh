# export CUDA_VISIBLE_DEVICES=0

SAVE_PATH=./checkpoints/mistal-7b-prm
MODEL=/nvme1/dyy/PRMDataSynthesis/main-train/checkpoints/Qwen2.5-Math-7B-PRM-Init
DATA_PATH=/nvme1/dyy/PRMDataSynthesis/main-train/datasets/ms-step

mkdir -p $SAVE_PATH

exec > >(tee ${SAVE_PATH}/output.log) 2>&1

deepspeed --module src.train_prm \
   --save_path $SAVE_PATH \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain $MODEL  \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset $DATA_PATH \
   --input_key question \
   --step_key steps \
   --step_label_key step_score \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
