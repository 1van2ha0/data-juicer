
import os
from typing import Dict, List, Optional, Any

from loguru import logger


from ..base_op import OPERATORS, UNFORKABLE, Mapper


@OPERATORS.register_module("replay_demos_mapper")
@UNFORKABLE.register_module("replay_demos_mapper")
class ReplayDemosMapper(Mapper):
    """
    Replay demonstrations with Isaac Lab environments and record videos.
    """

    _batched_op = True
    # Mark this operator as CUDA-accelerated
    _accelerator = "cuda"
    # Each task requires a new, clean Isaac Sim instance.
    _requires_actor_restart = True

    def __init__(
        self,
        # Task configuration
        task_name: Optional[str] = None,
        select_episodes: Optional[List[int]] = None,
        validate_states: bool = False,
        enable_pinocchio: bool = False,
        dual_arm: bool = False,
        device: str = "cuda:auto",

        # Input/Output keys in JSON metadata
        input_file_key: str = "dataset_file",
        output_file_key: str = "output_file",
        video_dir_key: str = "video_dir",

        # Video recording options
        video: bool = False,
        camera_view_list: Optional[List[str]] = None,
        save_depth: bool = False,

        # Isaac Sim options
        headless: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the ReplayDemosMapper.

        :param task_name: Isaac Lab task name (e.g., 'Isaac-Stack-Cube-Franka-IK-Rel-v0').
        :param select_episodes: A list of episode indices to be replayed.
            If None, replay all episodes.
        :param validate_states: Whether to validate states during replay.
        :param enable_pinocchio: Enable Pinocchio support.
        :param dual_arm: Whether the robot is a dual-arm robot.
        :param device: Device to run on ('cuda:0', 'cpu', etc.).
        :param input_file_key: Key in the sample to find the input HDF5 path.
        :param output_file_key: Key in the sample to store the output HDF5 path (if dumping).
        :param video_dir_key: Key in the sample to store the output video directory.
        :param video: Whether to record videos.
        :param camera_view_list: A list of camera views to record.
        :param save_depth: Whether to save depth images along with RGB.
        :param headless: Run Isaac Sim in headless mode.
        """
        super().__init__(*args, **kwargs)

        if video and not camera_view_list:
            raise ValueError("`camera_view_list` must be provided when `video` is True.")

        self.task_name = task_name
        self.select_episodes = select_episodes if select_episodes else []
        self.validate_states = validate_states
        self.enable_pinocchio = enable_pinocchio
        self.dual_arm = dual_arm
        self.device = device

        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.video_dir_key = video_dir_key

        self.video = video
        self.camera_view_list = camera_view_list if camera_view_list else []
        self.save_depth = save_depth
        self.headless = headless

        # Force batch_size=1 to ensure each actor processes exactly one task
        self.batch_size = 1

        # Lazy initialization for Isaac Sim
        self._env = None
        self._simulation_app = None
        self._isaac_initialized = False

        logger.info(f"Initialized ReplayDemosMapper for task={self.task_name}")

    def _ensure_sim_app(self):
        """Initialize Isaac Sim's SimulationApp once and reuse within the actor."""
        if self._isaac_initialized:
            return

        import argparse
        import sys
        import torch

        # If CUDA already initialized by torch, try to clear cache before launching App
        if torch.cuda.is_initialized():
            logger.warning("CUDA was initialized before Isaac Sim. Clearing cached state...")
            torch.cuda.empty_cache()

        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

        # Swap wrapped IO streams while launching SimulationApp (avoid wrapper issues)
        _orig_streams: Dict[str, Optional[Any]] = {}
        for stream_name in ("stdin", "stdout", "stderr"):
            _orig_streams[stream_name] = getattr(sys, stream_name, None)
            real_stream = getattr(sys, f"__{stream_name}__", None)
            if real_stream is not None:
                setattr(sys, stream_name, real_stream)

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args([])
        args.headless = self.headless
        args.device = self.device
        # enable cameras if video requested
        args.enable_cameras = bool(self.video)
        args.enable_scene_lights = not self.headless

        if self.enable_pinocchio:
            import pinocchio  # noqa: F401

        app_launcher = AppLauncher(args)
        self._simulation_app = app_launcher.app

        # Restore streams
        import sys as _sys

        for stream_name, orig_stream in _orig_streams.items():
            if orig_stream is not None:
                setattr(_sys, stream_name, orig_stream)

        # Ensure env packages registered
        import isaaclab_tasks  # noqa: F401

        self._isaac_initialized = True
        logger.info("Isaac Sim SimulationApp initialized for replay")

    def _create_env(self):
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # Build env config
        env_cfg = parse_env_cfg(self.task_name, device=self.device, num_envs=1)
        env_cfg.env_name = self.task_name

        # Extract success checking function and disable timeouts
        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

        # Some envs expect this
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.concatenate_terms = False

        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
            env_cfg.sim.physx.enable_ccd = True

        # Create environment
        env = gym.make(self.task_name, cfg=env_cfg).unwrapped

        self._env = env
        return success_term

    def _create_video_from_images(self, input_pattern: str, output_video: str, framerate: float = 20.0):
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(framerate),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "medium",
            output_video,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created video: {output_video}")
            return True
        except Exception as e:
            logger.warning(f"ffmpeg failed to create video {output_video}: {e}")
            return False

    def process_batched(self, samples, rank: Optional[int] = None):
        """Process a single replay task (batch_size=1)."""
        # Normalize device if auto and CUDA available
        try:
            import torch as _torch
            if isinstance(self.device, str) and self.device.startswith("cuda"):
                if self.device in ("cuda", "cuda:auto") and _torch.cuda.is_available():
                    count = _torch.cuda.device_count()
                    if count > 0:
                        idx = 0 if rank is None else rank % count
                        self.device = f"cuda:{idx}"
                elif not _torch.cuda.is_available():
                    logger.warning("CUDA requested but unavailable; falling back to CPU")
                    self.device = "cpu"
        except Exception:
            pass

        # Validate required input
        if self.input_file_key not in samples:
            logger.error("Missing required key '%s' in samples", self.input_file_key)
            samples.setdefault("replay_result", [None])
            samples["replay_result"][0] = {"success": False, "error": f"missing key {self.input_file_key}"}
            return samples

        # Only process first sample
        dataset_file = samples[self.input_file_key][0]
        # Optional overrides per-sample
        camera_views = samples.get("camera_view_list", [self.camera_view_list])[0] if "camera_view_list" in samples else self.camera_view_list
        save_depth = samples.get("save_depth", [self.save_depth])[0] if "save_depth" in samples else self.save_depth
        video_enabled = samples.get("video", [self.video])[0] if "video" in samples else self.video
        # Output base dir
        base_video_dir = samples.get(self.video_dir_key, [None])[0]
        if not base_video_dir:
            base_video_dir = os.path.join(os.getcwd(), f"{self.task_name}_videos")
        os.makedirs(base_video_dir, exist_ok=True)
        # Always allocate a unique sub-directory per task to avoid collisions across parallel tasks
        import time
        task_video_dir = os.path.join(base_video_dir, f"task_{os.getpid()}_{int(time.time()*1000)}")
        os.makedirs(task_video_dir, exist_ok=True)

        logger.info(
            "Replay task start: task=%s, dataset=%s, device=%s, video=%s, views=%s",
            self.task_name,
            dataset_file,
            self.device,
            video_enabled,
            camera_views,
        )

        # Results
        video_paths: List[str] = []
        failed_demo_ids: List[int] = []
        replayed_episode_count = 0

        try:
            # 1) Ensure SimulationApp
            self._ensure_sim_app()

            # 2) Create env
            success_term = self._create_env()

            # 3) Open dataset
            from isaaclab.utils.datasets import HDF5DatasetFileHandler
            dataset_handler = HDF5DatasetFileHandler()
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
            dataset_handler.open(dataset_file)
            episode_names = list(dataset_handler.get_episode_names())

            # If select_episodes provided, map from names by extracting indices
            if self.select_episodes:
                import re
                name_by_index: Dict[int, str] = {}
                for name in episode_names:
                    m = re.search(r"(\d+)", name)
                    if m:
                        name_by_index[int(m.group(1))] = name
                ordered = []
                for idx in self.select_episodes:
                    if idx in name_by_index:
                        ordered.append(name_by_index[idx])
                episode_names = ordered

            if len(episode_names) == 0:
                raise RuntimeError("No episodes found in dataset")

            env = self._env
            import torch
            # Default camera view if none provided
            if hasattr(env, "sim"):
                try:
                    env.sim.set_camera_view(eye=[3.0, 0.0, 1.5], target=[0.0, 0.0, 1.0])
                except Exception:
                    pass

            # Reset env
            env.reset()

            for name in episode_names:
                replayed_episode_count += 1
                # load episode (device-aware)
                episode = dataset_handler.load_episode(name, env.device)

                # Reset to initial state if available
                if "initial_state" in episode.data:
                    initial_state = episode.get_initial_state()
                    try:
                        env.sim.reset()
                        if hasattr(env, "recorder_manager"):
                            env.recorder_manager.reset()
                    except Exception:
                        pass
                    env.reset_to(initial_state, None, is_relative=True)

                # Prepare per-episode image save dir if video enabled
                if video_enabled and camera_views:
                    demo_save_dir = os.path.join(task_video_dir, "images", f"demo_{replayed_episode_count}")
                    os.makedirs(demo_save_dir, exist_ok=True)

                step_index = 0
                # Iterate actions
                while True:
                    next_action = episode.get_next_action()
                    if next_action is None:
                        break

                    # Suction support: last dim controls gripper, remaining are actions
                    action_tensor = torch.tensor(next_action, device=env.device)
                    if isinstance(action_tensor, torch.Tensor) and action_tensor.ndim == 1:
                        action_tensor = action_tensor.reshape(1, -1)

                    if "Suction" in self.task_name:
                        try:
                            if float(action_tensor[0, -1]) == 1.0:
                                env.open_suction_cup(0)
                            else:
                                env.close_suction_cup(0)
                            action_applied = action_tensor[:, :-1]
                        except Exception:
                            action_applied = action_tensor
                    else:
                        action_applied = action_tensor

                    env.step(action_applied)

                    # Save frames
                    if video_enabled and camera_views:
                        for view in camera_views:
                            try:
                                rgb_cam = (
                                    env.scene.sensors[f"{view}_cam"].data.output["rgb"].cpu().numpy()[0]
                                )
                                import cv2

                                rgb_path = os.path.join(
                                    demo_save_dir, f"frame_{step_index:04d}_{view}_rgb.png",
                                )
                                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_cam, cv2.COLOR_RGB2BGR))

                                if save_depth:
                                    depth_cam = (
                                        env.scene.sensors[f"{view}_cam"].data.output["distance_to_image_plane"].cpu().numpy()[0]
                                    )
                                    depth_16bit = (depth_cam * 1000).astype("uint16")
                                    depth_path = os.path.join(
                                        demo_save_dir, f"frame_{step_index:04d}_{view}_depth.png",
                                    )
                                    cv2.imwrite(depth_path, depth_16bit)
                            except Exception as e:
                                logger.debug(f"Failed saving frame for view {view}: {e}")

                    step_index += 1

                # Check success if term provided
                episode_success = True
                try:
                    success_term = success_term  # noqa
                    if success_term is not None:
                        result = success_term.func(env, **success_term.params)[0]
                        episode_success = bool(result)
                except Exception:
                    pass

                # Create videos per view
                if video_enabled and camera_views:
                    for view in camera_views:
                        # RGB video
                        input_pattern = os.path.join(task_video_dir, "images", f"demo_{replayed_episode_count}", f"frame_%04d_{view}_rgb.png")
                        output_video = os.path.join(task_video_dir, f"demo_{replayed_episode_count}_{view}_rgb.mp4")
                        ok = self._create_video_from_images(input_pattern, output_video)
                        if ok:
                            video_paths.append(output_video)

                        if save_depth:
                            input_pattern = os.path.join(task_video_dir, "images", f"demo_{replayed_episode_count}", f"frame_%04d_{view}_depth.png")
                            output_video = os.path.join(task_video_dir, f"demo_{replayed_episode_count}_{view}_depth.mp4")
                            ok = self._create_video_from_images(input_pattern, output_video)
                            if ok:
                                video_paths.append(output_video)

                # Record failure
                if not episode_success:
                    failed_demo_ids.append(replayed_episode_count)

            # Done
            result = {
                "success": True,
                "replayed_episode_count": replayed_episode_count,
                "failed_demo_ids": failed_demo_ids,
                "replay_video_paths": video_paths,
                "video_dir": task_video_dir,
            }
        except Exception as exc:
            logger.error("Replay task failed: %s", exc, exc_info=True)
            result = {"success": False, "error": str(exc)}
        finally:
            # Always cleanup env resources
            self.cleanup()

        # Populate standardized outputs for downstream aggregators
        samples.setdefault("replay_result", [None])
        samples["replay_result"][0] = result
        samples["replay_success"] = [bool(result.get("success", False))]
        samples["replay_video_paths"] = [result.get("replay_video_paths", [])]
        samples["replay_failed_demo_ids"] = [result.get("failed_demo_ids", [])]
        samples["replay_video_dir"] = [result.get("video_dir", base_video_dir)]
        samples["replay_failure_reason"] = [result.get("error", "") if not result.get("success", False) else ""]
        return samples

    def cleanup(self):
        # This will be implemented in the next step
        logger.info("Cleaning up ReplayDemosMapper resources...")
        if self._env is not None:
            self._env.close()
            self._env = None
        logger.info("Cleanup complete.")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
