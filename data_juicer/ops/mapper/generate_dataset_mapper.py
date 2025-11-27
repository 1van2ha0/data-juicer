import asyncio
import inspect
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from loguru import logger

from ..base_op import OPERATORS, UNFORKABLE, Mapper


@OPERATORS.register_module("generate_dataset_mapper")
@UNFORKABLE.register_module("generate_dataset_mapper")
class GenerateDatasetMapper(Mapper):
    """
    Generates a mimic dataset using Isaac Lab.

    This mapper integrates Isaac Lab's dataset generation pipeline to:
    1. Load annotated episodes from an HDF5 file.
    2. Use them as a source to generate a new mimic dataset in a simulation environment.
    3. Export the generated dataset to a new HDF5 file.

    Requires Isaac Lab environment to be properly installed.
    """

    _batched_op = True
    # Mark this operator as CUDA-accelerated
    _accelerator = "cuda"
    # Request actor restart after each task to ensure clean Isaac Sim state
    _requires_actor_restart = True

    def __init__(
        self,
        # Task configuration
        task_name: Optional[str] = None,
        num_envs: int = 8,
        generation_num_trials: int = 1000,
        device: str = "cuda:auto",

        # Input/Output keys in JSON metadata
        input_file_key: str = "input_file",
        output_file_key: str = "output_file",

        # Isaac Sim options
        headless: bool = True,
        enable_cameras: bool = False,
        enable_pinocchio: bool = False,
        pause_subtask: bool = False,

        *args,
        **kwargs,
    ):
        """
        Initialize the GenerateMimicDatasetMapper.

        :param task_name: Isaac Lab task name (required).
        :param num_envs: Number of parallel environments to use for generation.
        :param generation_num_trials: Number of trials for dataset generation.
        :param device: Device to run on ('cuda:0', 'cpu', etc.).
        :param input_file_key: Key in JSON for the input annotated HDF5 path.
        :param output_file_key: Key in JSON for the output generated HDF5 path.
    :param headless: Run Isaac Sim in headless mode.
    :param enable_cameras: Enable cameras in Isaac Sim.
    :param enable_pinocchio: Enable Pinocchio support for IK controllers.
    :param pause_subtask: Pause after every subtask during generation (debugging aid).
        """
        super().__init__(*args, **kwargs)
        
        if task_name is None:
            raise ValueError("task_name is required.")
            
        self.task_name = task_name
        self.num_envs = num_envs
        self.generation_num_trials = generation_num_trials
        # Honor user-provided runtime constraints; enable explicit num_proc handling like annotate mapper
        if getattr(self, "mem_required", 0) == 0 and getattr(self, "gpu_required", 0) == 0:
            self.mem_required = 1e-6

        self.device = device
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.headless = headless
        self.enable_cameras = enable_cameras
        self.enable_pinocchio = enable_pinocchio
        self.pause_subtask = pause_subtask

        # Lazy initialization
        self._env = None
        self._simulation_app = None
        self._isaac_initialized = False
        self._success_term = None
        self._orig_streams: Dict[str, Optional[Any]] = {}
        self._faulthandler_file = None
        self._output_file_abs: Optional[str] = None
        
        # Force batch_size=1 to ensure each actor processes exactly one task
        self.batch_size = 1
        
        logger.info(
            f"Initialized GenerateMimicDatasetMapper: task={self.task_name}, "
            f"num_envs={self.num_envs}, batch_size=1 (one task per actor)"
        )

    def _ensure_sim_app(self):
        """Ensure Isaac Sim SimulationApp is initialized once and reused."""
        if self._isaac_initialized:
            return

        import argparse
        import faulthandler
        import sys
        import torch

        if torch.cuda.is_initialized():
            logger.warning("CUDA was initialized before Isaac Sim. Clearing cached state...")
            torch.cuda.empty_cache()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Isaac Sim requires CUDA to run. "
                "Please verify the GPU driver and CUDA installation."
            )

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        logger.info(
            "Initializing Isaac Sim SimulationApp for task %s (device=%s, headless=%s)",
            self.task_name,
            self.device,
            self.headless,
        )
        logger.info("CUDA device count detected by torch: %d", torch.cuda.device_count())

        # Swap wrapped IO streams with the real OS streams while launching SimulationApp
        for stream_name in ('stdin', 'stdout', 'stderr'):
            self._orig_streams[stream_name] = getattr(sys, stream_name, None)
            real_stream = getattr(sys, f"__{stream_name}__", None)
            if real_stream is not None:
                setattr(sys, stream_name, real_stream)

        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args([])
        args.headless = self.headless
        args.device = self.device
        args.enable_cameras = self.enable_cameras
        args.enable_scene_lights = not self.headless

        if self.enable_pinocchio:
            import pinocchio  # noqa: F401

        try:
            self._faulthandler_file = open(os.devnull, "w")
            faulthandler.enable(file=self._faulthandler_file)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.debug("Failed to enable faulthandler when launching SimulationApp: %s", exc)
            self._faulthandler_file = None

        app_launcher = AppLauncher(args)
        self._simulation_app = app_launcher.app

        # Restore wrapped streams
        for stream_name, orig_stream in self._orig_streams.items():
            if orig_stream is not None:
                setattr(sys, stream_name, orig_stream)

        import isaaclab_mimic.envs  # noqa: F401
        if self.enable_pinocchio:
            import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

        self._isaac_initialized = True
        logger.info("Isaac Sim SimulationApp initialized successfully")

    def _resolve_env_name(self, input_file: str) -> str:
        if self.task_name:
            return self.task_name.split(":")[-1]

        from isaaclab_mimic.datagen.utils import get_env_name_from_dataset

        return get_env_name_from_dataset(input_file)

    def _create_task_env(self, input_file: str, output_file: str) -> Optional[Any]:
        import gymnasium as gym
        import omni
        import torch
        from isaaclab.envs import ManagerBasedRLMimicEnv
        from isaaclab_mimic.datagen.generation import setup_env_config
        from isaaclab_mimic.datagen.utils import setup_output_paths

        if self._env is not None:
            self._env.close()
            self._env = None

        env_name = self._resolve_env_name(input_file)
        output_dir, output_filename = setup_output_paths(output_file)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            "Preparing Isaac Lab environment for dataset generation: env=%s, num_envs=%d, trials=%d",
            env_name,
            self.num_envs,
            self.generation_num_trials,
        )

        env_cfg, success_term = setup_env_config(
            env_name=env_name,
            output_dir=output_dir,
            output_file_name=output_filename,
            num_envs=self.num_envs,
            device=self.device,
            generation_num_trials=self.generation_num_trials,
        )

        self._env = gym.make(env_name, cfg=env_cfg).unwrapped

        if not isinstance(self._env, ManagerBasedRLMimicEnv):
            raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

        if "action_noise_dict" not in inspect.signature(self._env.target_eef_pose_to_action).parameters:
            omni.log.warn(
                f'The "noise" parameter in the "{env_name}" environment\'s mimic API '
                '"target_eef_pose_to_action" is deprecated. Please update the API to take '
                "action_noise_dict instead."
            )

        if hasattr(self._env, "cfg") and hasattr(self._env.cfg, "datagen_config"):
            seed = getattr(self._env.cfg.datagen_config, "seed", None)
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

        self._env.reset()
        self._success_term = success_term
        self._output_file_abs = os.path.join(output_dir, output_filename)
        return success_term

    def _run_async_generation(self, input_file: str, success_term: Optional[Any]) -> None:
        from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation

        async_components = setup_async_generation(
            env=self._env,
            num_envs=self.num_envs,
            input_file=input_file,
            success_term=success_term,
            pause_subtask=self.pause_subtask,
        )

        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        try:
            env_loop(
                self._env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            logger.warning("Async generation tasks were cancelled early.")
        finally:
            data_gen_tasks.cancel()
            try:
                async_components["event_loop"].run_until_complete(data_gen_tasks)
            except asyncio.CancelledError:
                logger.info("Cancelled async tasks cleaned up successfully.")
            except Exception as loop_exc:  # pragma: no cover - best effort cleanup logging
                logger.warning("Exception while finalizing async tasks: %s", loop_exc)

    def _generate_dataset_for_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Generates mimic dataset from a single annotated source file."""
        self._ensure_sim_app()
        success_term = self._create_task_env(input_file, output_file)
        self._run_async_generation(input_file, success_term)

        output_path = self._output_file_abs or output_file
        success = output_path is not None and os.path.exists(output_path)

        if success:
            logger.info("Dataset generation completed. Output file located at %s", output_path)
        else:
            logger.warning(
                "Expected output dataset %s was not found after generation. "
                "Downstream steps should verify dataset export.",
                output_path,
            )

        return {
            "success": success,
            "output_file": output_path,
            "exported_episodes": None,
        }

    def process_batched(self, samples: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single dataset generation task (batch_size=1).
        """
        num_samples = len(samples.get(self.text_key, []))
        if num_samples != 1:
            logger.warning(
                "GenerateMimicDatasetMapper expects batch_size=1, received %d. Only the first sample will be processed.",
                num_samples,
            )
            samples = {key: [values[0]] for key, values in samples.items() if values}

        try:
            import os as _os
            import torch as _torch

            if isinstance(self.device, str) and self.device.startswith("cuda"):
                if self.device in ("cuda", "cuda:auto"):
                    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if visible and len(visible.split(",")) == 1:
                        self.device = "cuda:0"
                    elif _torch.cuda.is_available():
                        count = _torch.cuda.device_count()
                        if count > 0:
                            idx = 0 if rank is None else rank % count
                            self.device = f"cuda:{idx}"
                elif not _torch.cuda.is_available():
                    logger.warning("Requested device %s but CUDA is unavailable; falling back to CPU", self.device)
                    self.device = "cpu"
        except Exception:  # pragma: no cover - defensive
            pass

        logger.info("Processing single generation task on device %s", self.device)

        results: List[Optional[Dict[str, Any]]] = [None]

        if self.input_file_key not in samples or self.output_file_key not in samples:
            logger.error(
                "Required keys '%s' or '%s' not found in samples.",
                self.input_file_key,
                self.output_file_key,
            )
            samples["generation_result"] = results
            return samples

        input_file = samples[self.input_file_key][0]
        output_file = samples[self.output_file_key][0]

        logger.info("Task: Generate from %s -> %s", input_file, output_file)

        try:
            result = self._generate_dataset_for_file(input_file, output_file)
            results[0] = result

            if result.get("success"):
                logger.info("✓ Task completed successfully")
            else:
                logger.warning("✗ Task failed")

        except Exception as exc:
            logger.error("An exception occurred during dataset generation: %s", exc, exc_info=True)
            results[0] = {"success": False, "error": str(exc)}
        finally:
            logger.info("Task complete. Cleaning up resources...")
            self.cleanup()

        samples["generation_result"] = results
        return samples

    def cleanup(self):
        """
        Cleanup Isaac Sim resources (called by the framework).

        Only closes the Isaac Lab environment, not the simulation app.
        Ray will manage the Actor process lifecycle.
        """
        logger.info("Cleaning up GenerateMimicDatasetMapper resources...")
        
        if self._env is not None:
            self._env.close()
            logger.info("Closed Isaac Lab environment.")
            self._env = None
        
        self._success_term = None
        self._output_file_abs = None

        if self._faulthandler_file is not None:
            try:
                import faulthandler

                faulthandler.disable()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            try:
                self._faulthandler_file.close()
            except Exception:
                pass
            finally:
                self._faulthandler_file = None

        logger.info("Cleanup complete (simulation_app left for Ray to manage).")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
