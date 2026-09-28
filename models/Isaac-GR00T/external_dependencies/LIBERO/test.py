from libero.libero import benchmark
from libero.libero.utils import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 0
task = task_suite.get_task(task_id)
task_description = task.language

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

print(f"[info] task: {task_description}", flush=True)
print(f"[info] bddl: {task_bddl_file}", flush=True)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

print("[debug] creating env...", flush=True)
env = OffScreenRenderEnv(**env_args)
print("[debug] env created", flush=True)

print("[debug] seed...", flush=True)
env.seed(0)

print("[debug] reset...", flush=True)
env.reset()
print("[debug] reset done", flush=True)

print("[debug] loading init states...", flush=True)
init_states = task_suite.get_task_init_states(task_id)
print(f"[debug] init states loaded: {len(init_states)}", flush=True)

init_state_id = 0
print("[debug] setting init state...", flush=True)
env.set_init_state(init_states[init_state_id])
print("[debug] init state set", flush=True)

dummy_action = [0.0] * 7

for step in range(10):
    print(f"[debug] step {step}", flush=True)
    obs, reward, done, info = env.step(dummy_action)
    print(f"[debug] step {step}, reward={reward}, done={done}", flush=True)

env.close()
print("[debug] env closed, test ok", flush=True)