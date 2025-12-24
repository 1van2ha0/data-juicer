import json
import shutil
import subprocess
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from ..base_op import OPERATORS, UNFORKABLE, Mapper


@OPERATORS.register_module("convert_to_lerobot_mapper")
@UNFORKABLE.register_module("convert_to_lerobot_mapper")
class ConvertToLeRobotMapper(Mapper):
    """
    Convert HDF5 datasets (MimicGen/Isaac Lab format) to LeRobot dataset format.
    """

    _batched_op = True

    def __init__(
        self,
        config_path: str = None,
        input_file_key: str = "input_file",
        output_dir_key: str = "output_dir",
        video_dir_key: str = "video_dir",
        config_path_key: str = "config_path",
        *args,
        **kwargs,
    ):
        kwargs["ray_execution_mode"] = "task"
        super().__init__(*args, **kwargs)

        self.config_path = config_path
        self.input_file_key = input_file_key
        self.output_dir_key = output_dir_key
        self.video_dir_key = video_dir_key
        self.config_path_key = config_path_key

        # Force batch_size=1 to handle one dataset conversion per process
        self.batch_size = 1

        # Load configuration template if provided
        if config_path:
            self.base_config = self._load_config(config_path)
        else:
            self.base_config = None

    def _load_config(self, config_path):
        if config_path is None:
            raise ValueError("config_path is required.")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            external_config = yaml.safe_load(f)

        # Create a config dict
        config = {}
        config["external_config"] = external_config

        # Defaults
        config["chunks_size"] = 1000
        config["fps"] = 20
        config["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        config["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        config["modality_fname"] = "modality.json"
        config["episodes_fname"] = "episodes.jsonl"
        config["tasks_fname"] = "tasks.jsonl"
        config["info_fname"] = "info.json"
        config["task_index"] = 0
        config["total_episodes"] = 5000

        # Override from YAML
        if "dataset" in external_config:
            ds_config = external_config["dataset"]
            for field_name in ["robot_type", "fps", "chunks_size"]:
                if field_name in ds_config:
                    config[field_name] = ds_config[field_name]

        # Handle templates
        # Resolve relative paths relative to the config file location if possible, or workspace root
        # Here we assume they are relative to workspace root if not absolute
        if "modality_template_path" in external_config:
            config["modality_template_path"] = Path(external_config["modality_template_path"])
        if "info_template_path" in external_config:
            config["info_template_path"] = Path(external_config["info_template_path"])

        if "tasks" in external_config:
            config["tasks"] = external_config["tasks"]

        return config

    def _get_video_metadata(self, video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=height,width,codec_name,pix_fmt,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]

        try:
            output = subprocess.check_output(cmd).decode("utf-8")
            probe_data = json.loads(output)
            stream = probe_data["streams"][0]

            # Parse frame rate
            if "/" in stream["r_frame_rate"]:
                num, den = map(int, stream["r_frame_rate"].split("/"))
                fps = num / den
            else:
                fps = float(stream["r_frame_rate"])

            # Check for audio streams
            audio_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(video_path),
            ]
            audio_output = subprocess.check_output(audio_cmd).decode("utf-8")
            audio_data = json.loads(audio_output)
            has_audio = len(audio_data.get("streams", [])) > 0

            metadata = {
                "dtype": "video",
                "shape": [stream["height"], stream["width"], 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": stream["codec_name"],
                    "video.pix_fmt": stream["pix_fmt"],
                    "video.is_depth_map": False,
                    "has_audio": has_audio,
                },
            }
            return metadata

        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"Error getting metadata for {video_path}: {e}")
            return None

    def _get_feature_info(self, step_data: pd.DataFrame, video_paths: dict, config: dict) -> dict:
        features = {}
        for video_key, video_path in video_paths.items():
            video_metadata = self._get_video_metadata(video_path)
            if video_metadata:
                features[video_key] = video_metadata

        lerobot_keys = config["external_config"].get("lerobot_keys", {})
        state_key = lerobot_keys.get("state", "observation.state")
        action_key = lerobot_keys.get("action", "action")

        for column in step_data.columns:
            column_data = np.stack(step_data[column], axis=0)
            shape = column_data.shape
            if len(shape) == 1:
                shape = (1,)
            else:
                shape = shape[1:]
            features[column] = {
                "dtype": column_data.dtype.name,
                "shape": shape,
            }
            # State & action
            if column in [state_key, action_key]:
                dof = column_data.shape[1]
                features[column]["names"] = [f"motor_{i}" for i in range(dof)]

        return features

    def _generate_info(
        self,
        total_episodes: int,
        total_frames: int,
        total_tasks: int,
        total_videos: int,
        total_chunks: int,
        config: dict,
        step_data: pd.DataFrame,
        video_paths: dict,
    ) -> dict:
        with open(config["info_template_path"]) as fp:
            info_template = json.load(fp)

        info_template["robot_type"] = config.get("robot_type")
        info_template["total_episodes"] = total_episodes
        info_template["total_frames"] = total_frames
        info_template["total_tasks"] = total_tasks
        info_template["total_videos"] = total_videos
        info_template["total_chunks"] = total_chunks
        info_template["chunks_size"] = config["chunks_size"]
        info_template["fps"] = config["fps"]

        info_template["data_path"] = config["data_path"]
        info_template["video_path"] = config["video_path"]

        features = self._get_feature_info(step_data, video_paths, config=config)
        info_template["features"] = features

        return info_template

    def _parse_structured_slice(self, slice_config):
        """
        Parses a structured slice configuration from YAML (list of dims) into a slice object or tuple.
        Example YAML: [[null, -1], 0] -> Python: (slice(None, -1), 0)
        """
        if not isinstance(slice_config, list):
            raise ValueError(f"Slice config must be a list of dimensions, got {type(slice_config)}")

        slices = []
        for dim in slice_config:
            if isinstance(dim, int):
                # Integer index
                slices.append(dim)
            elif isinstance(dim, str) and dim == "...":
                # Ellipsis
                slices.append(Ellipsis)
            elif isinstance(dim, list):
                # Slice definition [start, stop, step]
                # YAML null becomes Python None
                slices.append(slice(*dim))
            else:
                raise ValueError(f"Invalid slice dimension format: {dim}")
        
        if len(slices) == 1:
            return slices[0]
        return tuple(slices)

    def _apply_transform(self, array: np.ndarray, transform_config: dict) -> np.ndarray:
        if "slice" in transform_config:
            slice_conf = transform_config["slice"]
            try: 
                slice_obj = self._parse_structured_slice(slice_conf)
                array = array[slice_obj]
            except Exception as e:
                logger.error(f"Error applying slice {slice_conf}: {e}")
                raise

        if "reshape" in transform_config:
            shape = transform_config["reshape"]
            array = array.reshape(shape)

        return array

    def _convert_trajectory_to_df(
        self,
        trajectory: h5py.Group,
        episode_index: int,
        index_start: int,
        config: dict,
    ) -> dict:
        return_dict = {}
        data = {}

        mapping = config["external_config"].get("mapping", {})
        lerobot_keys = config["external_config"].get("lerobot_keys", {})

        state_key = lerobot_keys.get("state", "observation.state")
        action_key = lerobot_keys.get("action", "action")
        annotation_keys = lerobot_keys.get(
            "annotation", ["annotation.human.action.task_description", "annotation.human.action.valid"]
        )

        # 1. Get state and action
        for key_type in ["state", "action"]:
            if key_type not in mapping:
                continue

            lerobot_key_name = state_key if key_type == "state" else action_key

            concatenated_list = []
            for source_config in mapping[key_type]:
                hdf5_key = source_config["key"]
                key_path = hdf5_key.split(".")
                try:
                    array = reduce(lambda x, y: x[y], key_path, trajectory)
                    array = np.array(array).astype(np.float64)
                    array = self._apply_transform(array, source_config)
                    concatenated_list.append(array)
                except KeyError:
                    logger.warning(f"Key {hdf5_key} not found in trajectory")
                    continue

            if concatenated_list:
                concatenated = np.concatenate(concatenated_list, axis=1)
                data[lerobot_key_name] = [row for row in concatenated]

        if action_key not in data or state_key not in data:
            raise ValueError(f"Missing state or action data. Keys found: {list(data.keys())}")

        assert len(data[action_key]) == len(data[state_key])
        length = len(data[action_key])
        data["timestamp"] = np.arange(length).astype(np.float64) * (1.0 / config["fps"])

        # 2. Get the annotation
        data[annotation_keys[0]] = np.ones(length, dtype=int) * config["task_index"]
        data[annotation_keys[1]] = np.ones(length, dtype=int) * 1

        # 3. Other data
        data["episode_index"] = np.ones(length, dtype=int) * episode_index
        data["task_index"] = np.zeros(length, dtype=int)
        data["index"] = np.arange(length, dtype=int) + index_start

        reward = np.zeros(length, dtype=np.float64)
        reward[-1] = 1
        done = np.zeros(length, dtype=bool)
        done[-1] = True
        data["next.reward"] = reward
        data["next.done"] = done

        dataframe = pd.DataFrame(data)

        return_dict["data"] = dataframe
        return_dict["length"] = length
        return_dict["annotation"] = set(data[annotation_keys[0]]) | set(data[annotation_keys[1]])
        return return_dict

    def _check_failed_videos(self, video_dir: Path) -> list:
        if not video_dir.exists():
            logger.warning(f"Video directory not found: {video_dir}")
            return []

        video_files = list(video_dir.glob("*.mp4"))
        failed_ids = []
        for video_file in video_files:
            if "failed" in video_file.name:
                try:
                    traj_id = video_file.name.split("_")[1]
                    if traj_id not in failed_ids:
                        failed_ids.append(traj_id)
                except IndexError:
                    pass
        return failed_ids

    def _convert_file(self, input_file: str, output_dir: str, video_dir: str = None, config_path: str = None):
        # Determine config to use
        if config_path:
            config = self._load_config(config_path)
        elif self.base_config:
            config = self.base_config.copy()
        else:
            logger.error("No configuration provided for file conversion.")
            return

        input_path = Path(input_file)
        output_path = Path(output_dir)

        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            return

        # Resolve video_dir
        if video_dir:
            video_dir = Path(video_dir)
            if not video_dir.is_absolute():
                # If relative, assume relative to input file's directory
                video_dir = input_path.parent / video_dir
        else:
            # Fallback or assume it's relative to input file location
            video_dir = input_path.parent / "videos"  # Assumption

        # Validate templates
        if "modality_template_path" not in config or not config["modality_template_path"].exists():
            logger.error(f"Modality template not found: {config.get('modality_template_path')}")
            return
        if "info_template_path" not in config or not config["info_template_path"].exists():
            logger.error(f"Info template not found: {config.get('info_template_path')}")
            return

        hdf5_handler = h5py.File(input_path, "r")
        hdf5_data = hdf5_handler["data"]

        # Prepare output directories
        output_path.mkdir(parents=True, exist_ok=True)
        lerobot_meta_dir = output_path / "meta"
        lerobot_meta_dir.mkdir(parents=True, exist_ok=True)

        total_length = 0
        example_data = None
        video_paths = {}

        trajectory_ids = sorted(
            [k for k in hdf5_data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1])
        )

        failed_ids = self._check_failed_videos(video_dir)

        episodes_info = []
        logger.info(f"Processing {len(trajectory_ids)} trajectories from {input_file}...")

        for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids)):
            if trajectory_id in [f"demo_{failed_id}" for failed_id in failed_ids]:
                continue

            trajectory = hdf5_data[trajectory_id]

            try:
                df_ret_dict = self._convert_trajectory_to_df(
                    trajectory=trajectory, episode_index=episode_index, index_start=total_length, config=config
                )
            except Exception as e:
                logger.error(f"Failed to convert trajectory {trajectory_id}: {e}")
                continue

            # Save episode data
            dataframe = df_ret_dict["data"]
            episode_chunk = episode_index // config["chunks_size"]
            save_relpath = config["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
            save_path = output_path / save_relpath
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(save_path)

            length = df_ret_dict["length"]
            total_length += length
            episodes_info.append(
                {
                    "episode_index": episode_index,
                    "tasks": [config["tasks"][task_index] for task_index in df_ret_dict["annotation"]],
                    "length": length,
                }
            )

            # Process videos
            video_mapping = config["external_config"].get("video_mapping", {})
            for lerobot_key, view_suffix in video_mapping.items():
                try:
                    new_video_relpath = config["video_path"].format(
                        episode_chunk=episode_chunk, video_key=lerobot_key, episode_index=episode_index
                    )
                    new_video_path = output_path / new_video_relpath
                    new_video_path.parent.mkdir(parents=True, exist_ok=True)

                    original_video_path = video_dir / f"{trajectory_id}_{view_suffix}.mp4"
                    if original_video_path.exists():
                        shutil.copy2(original_video_path, new_video_path)
                        if lerobot_key not in video_paths:
                            video_paths[lerobot_key] = new_video_path
                    else:
                        logger.warning(f"Video file not found: {original_video_path}")
                except Exception as e:
                    logger.warning(f"Error processing video {lerobot_key} for {trajectory_id}: {e}")

            if example_data is None:
                example_data = df_ret_dict

            if len(episodes_info) > config["total_episodes"] - 1:
                break

        # Generate meta files
        tasks_path = lerobot_meta_dir / config["tasks_fname"]
        task_jsonlines = [{"task_index": task_index, "task": task} for task_index, task in config["tasks"].items()]
        with open(tasks_path, "w") as f:
            for item in task_jsonlines:
                f.write(json.dumps(item) + "\n")

        episodes_path = lerobot_meta_dir / config["episodes_fname"]
        with open(episodes_path, "w") as f:
            for item in episodes_info:
                f.write(json.dumps(item) + "\n")

        modality_path = lerobot_meta_dir / config["modality_fname"]
        shutil.copy(config["modality_template_path"], modality_path)

        if example_data:
            info_json = self._generate_info(
                total_episodes=len(episodes_info),
                total_frames=total_length,
                total_tasks=len(config["tasks"]),
                total_videos=len(episodes_info),
                total_chunks=(episode_index + config["chunks_size"] - 1) // config["chunks_size"],
                step_data=example_data["data"],
                video_paths=video_paths,
                config=config,
            )
            with open(lerobot_meta_dir / "info.json", "w") as f:
                json.dump(info_json, f, indent=4)

        hdf5_handler.close()
        logger.info(f"Conversion completed for {input_file}")

    def process_batched(self, samples: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
        input_files = samples[self.input_file_key]
        output_dirs = samples[self.output_dir_key]

        if self.video_dir_key in samples:
            video_dirs = samples[self.video_dir_key]
        else:
            video_dirs = [None] * len(input_files)

        if self.config_path_key in samples:
            config_paths = samples[self.config_path_key]
        else:
            config_paths = [None] * len(input_files)

        for input_file, output_dir, video_dir, config_path in zip(input_files, output_dirs, video_dirs, config_paths):
            try:
                self._convert_file(input_file, output_dir, video_dir=video_dir, config_path=config_path)
            except Exception as e:
                logger.exception(f"Failed to convert {input_file}: {e}")

        return samples
