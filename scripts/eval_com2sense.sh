TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="roberta-base"


python3 -m trainers.eval.py \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --gradient_accumulation_steps 4 \
  --per_gpu_eval_batch_size 796 \
  --learning_rate 1e-5 \
  --model_name_or_path outputs/com2sense/ckpts/ckpts/checkpoint-8000
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 100 \
  --logging_steps 2000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --iters_to_eval 10000 \
  # --max_eval_steps 1000 \
