#!/usr/bin/env python3

import os
import ray
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper import AnnotateDemosMapper

ray.init(address='local')

ds = RayDataset(ray.data.read_json('./demos/franka_mimicgen/annotate/annotation_tasks.jsonl'))

annotator = AnnotateDemosMapper(
    task_name='Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0',
    num_proc=2,
    gpu_required=0.5,    
    accelerator='cuda',
    headless=True,
    enable_cameras=True,
    enable_pinocchio=False,
    batch_size=1,
    num_cpus=1
)

ds = ds.process([annotator])

output_path = './outputs/demo/mimicgen/annotate'
os.makedirs(output_path, exist_ok=True)
ds.data.write_json(output_path, force_ascii=False)

ray.shutdown()
