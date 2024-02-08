# single-scale training and multi-scale testing setting proposed in mip-splatting

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]#remove flowers, treehill
scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room"]
factors = [1, 1, 1, 1, 1, 1, 1]

excluded_gpus = set([])

output_dir = "360v2_ours_stmt_down_resize"

dry_run = False

jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):
    get_folder = "/cluster/work/cvl/jiezcao/jiameng/mip-splatting/benchmark_360v2_ours_stmt_down/"
    trained_gaussian = os.path.join(get_folder, scene, "point_cloud/iteration_30000/point_cloud.ply")# "./fused/"+scene+"_fused_x1.ply"
    for scale in [8, 4, 2, 1]:
        pseudo_gt = os.path.join(get_folder, scene, "pseudo_gt/resize_x" + str(scale))
        print(pseudo_gt)
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train_two_stage.py -s {pseudo_gt} -m {output_dir}/{scene} -r 1 --port {5009 + int(gpu)} --kernel_size 0.1 --load_gaussian {trained_gaussian}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render_ours.py -m {output_dir}/{scene} -r 1 --data_device cpu --skip_train"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
