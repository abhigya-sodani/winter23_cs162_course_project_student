TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="roberta-base"


python3 -m trainers.train \
  --model_name_or_path outputs/com2sense/ckpts/ckpts/checkpoint-8000 \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 796 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 100 \
  --logging_steps 1 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --overwrite_output_dir \
  # --max_eval_steps 1000 \
