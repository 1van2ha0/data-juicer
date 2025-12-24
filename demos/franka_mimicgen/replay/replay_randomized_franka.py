#!/usr/bin/env python3

import os
import ray
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper import ReplayDemosRandomizedMapper

ray.init(address='auto')

ds = RayDataset(ray.data.read_json('./demos/franka_mimicgen/replay/replay_tasks.jsonl'))

replayer = ReplayDemosRandomizedMapper(
    task_name='Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0',
    headless=True,
    device='cpu',
    video=True,
    camera_view_list=['table', 'wrist'],
    save_depth=False,
    num_proc=2,
    gpu_required=0.5,
    input_file_key='dataset_file',
    video_dir_key='video_dir',
    visual_randomization_config='./demos/franka_mimicgen/replay/visual_randomization.yaml',
    accelerator='cuda',
    batch_size=1
)

ds = ds.process([replayer])

output_path = './outputs/demo/mimicgen/replay'
os.makedirs(output_path, exist_ok=True)
ds.data.write_json(output_path, force_ascii=False)

ray.shutdown()
