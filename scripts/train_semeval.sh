TASK_NAME="semeval"
DATA_DIR="datasets/semeval_2020_task4"
MODEL_TYPE="microsoft/deberta-base"


python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 796 \
  --learning_rate 1e-5 \
  --max_steps 0 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 1000 \
  --logging_steps 2000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --iters_to_eval 1000 \
  --overwrite_output_dir \
  # --max_eval_steps 1000 \
