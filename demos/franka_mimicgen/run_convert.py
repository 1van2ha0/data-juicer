#!/usr/bin/env python3

import os
import ray
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper import ConvertToLeRobotMapper

# NOTE: Before using, please check which video directory is specified in the `video_dir_key`. You can open the `./outputs/demo/mimicgen/replay/videos_run{x}` directory to see the specific video folder name, and then modify the video_dir field in convert_tasks.jsonl to match the correct path.

ray.init(address='local')

# Updated path to the new data directory
ds = RayDataset(ray.data.read_json('./demos/franka_mimicgen/data/convert_tasks.jsonl'))

converter = ConvertToLeRobotMapper(
    config_path_key='config_path',
    input_file_key='input_file',
    output_dir_key='output_dir',
    video_dir_key='video_dir',
    num_proc=2
)

ds = ds.process([converter])

output_path = './outputs/demo/mimicgen/convert'
os.makedirs(output_path, exist_ok=True)
ds.data.write_json(output_path, force_ascii=False)

ray.shutdown()
