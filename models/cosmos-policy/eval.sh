GPUS=0,1,2,3,4,5,6,7 MAX_PARALLEL_JOBS=8 ./run_libero_3seeds.sh


setsid env \
    GPUS=0,2,5,6 \
    MAX_PARALLEL_JOBS=4 \
    RUN_NAME=libero_allgpu_run \
    ./run_libero_3seeds_multigpu.sh \
    > libero_allgpu_run.log 2>&1 < /dev/null &


