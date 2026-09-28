GPUS="0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7" \
  EVAL_WORLD_SIZE=16 \
  RUN_NAME=libero_plus_16workers_auto \
  MODEL_NAME=UnifoLM-VLA \
  bash scripts/eval_scripts/eval_libero_plus.sh \
    /mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt \
    /mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLM-Base \
    8