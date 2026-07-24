ONDA_ENV=/root/envs/gr00t_libero MODEL_PATH=/mnt/afs/zhengmingkai/raozf/models/Gr00t-N1.7-libero/libero_spatial  COSMOS_PATH=/mnt/afs/zhengmingkai/raozf/models/Cosmos-Reason2-2B SUITE=libero_spatial CUDA_VISIBLE_DEVICES=0 N_EPISODES=150 N_ENVS=5 N_ACTION_STEPS=8 MAX_EPISODE_STEPS=720 ONLY_FIRST_TASK=1 bash run_libero_closed_loop_conda.sh



SUITE=libero_goal ./eval_libero_plus_suite.sh