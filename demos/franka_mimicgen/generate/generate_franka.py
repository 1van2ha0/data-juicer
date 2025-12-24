#!/usr/bin/env python3

import os
import ray
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper import GenerateDatasetMapper

ray.init(address='local')

ds = RayDataset(ray.data.read_json('./demos/franka_mimicgen/generate/generate_tasks.jsonl'))

generator = GenerateDatasetMapper(
    task_name='Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0',
    num_envs=8,
    generation_num_trials=10,
    device='cuda:auto',
    headless=True,
    enable_cameras=True,
    input_file_key='input_file',
    output_file_key='output_file',
    num_proc=2,
    gpu_required=0.5,
    accelerator='cuda',
    batch_size=1
)

ds = ds.process([generator])

output_path = './outputs/demo/mimicgen/generate'
os.makedirs(output_path, exist_ok=True)
ds.data.write_json(output_path, force_ascii=False)

ray.shutdown()
