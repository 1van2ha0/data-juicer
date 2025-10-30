import os
from typing import Any, Dict, List, Optional

from loguru import logger

from ..base_op import OPERATORS, UNFORKABLE, Simulator


@OPERATORS.register_module("annotate_demos_mapper")
@UNFORKABLE.register_module("annotate_demos_mapper")
class AnnotateDemosMapper(Simulator):
    """
    Automatically annotate robot demonstration episodes using Isaac Lab.
    
    This mapper integrates Isaac Lab's automatic annotation pipeline to:
    1. Load episodes from HDF5 files
    2. Replay them in Isaac Sim environment
    3. Automatically detect subtask completions
    4. Export annotated episodes with subtask term signals
    
    Requires Isaac Lab environment to be properly installed.
    """

    _batched_op = True
    # Mark this operator as CUDA-accelerated so the framework passes rank and sets proper mp start method
    _accelerator = "cuda"
    # Request actor restart after each task to ensure clean Isaac Sim state
    _requires_actor_restart = True

    def __init__(
        self,
        # Task configuration
        task_name: Optional[str] = None,
        device: str = "cuda:auto",
        
        # Input/Output keys in JSON metadata
        input_file_key: str = "input_file",
        output_file_key: str = "output_file",
        
        
        # Isaac Sim options
        headless: bool = True,
        enable_cameras: bool = False,
        enable_pinocchio: bool = False,
        
        *args,
        **kwargs,
    ):
        """
        Initialize the AnnotateDemosMapper.
        
        Each sample in the JSON should contain:
        - input_file: Path to input HDF5 file (with all episodes)
        - output_file: Path to output annotated HDF5 file
        
        For distributed processing, create a JSONL with multiple tasks:
        {"text": "task1", "input_file": "file1.hdf5", "output_file": "file1_ann.hdf5"}
        {"text": "task2", "input_file": "file2.hdf5", "output_file": "file2_ann.hdf5"}
        ...
        
        Data-Juicer will distribute these tasks across workers.
        
        :param task_name: Isaac Lab task name (required)
        :param device: Device to run on ('cuda:0', 'cpu', etc.)
        :param input_file_key: Key in JSON containing input HDF5 path
        :param output_file_key: Key in JSON containing output HDF5 path
        :param headless: Run Isaac Sim in headless mode
        :param enable_cameras: Enable cameras in Isaac Sim
        :param enable_pinocchio: Enable Pinocchio for IK controllers
        """
        super().__init__(*args, **kwargs)
        # By default, Data-Juicer will cap runtime_np to GPU count when both
        # mem_required and gpu_required are 0. For this operator we honor user-specified
        # num_proc even if it exceeds GPU count (the user controls oversubscription),
        # so we set a tiny non-zero mem_required to bypass that auto-cap unless the
        # user explicitly provided mem_required/gpu_required.
        if getattr(self, "mem_required", 0) == 0 and getattr(self, "gpu_required", 0) == 0:
            # store in GB (consistent with base OP conversion), ~1KB
            self.mem_required = 1e-6
        
        if task_name is None:
            raise ValueError(
                "task_name is required. Please set it in your config, e.g.:\n"
                "process:\n"
                "  - annotate_demos_mapper:\n"
                "      task_name: 'Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0'"
            )
        self.task_name = task_name
        self.device = device
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.headless = headless
        self.enable_cameras = enable_cameras
        self.enable_pinocchio = enable_pinocchio
        
        # Lazy initialization - will be set on first use
        self._env = None
        self._success_term = None
        self._simulation_app = None
        # file handle used to enable faulthandler safely
        self._faulthandler_file = None
        # original std streams to restore on cleanup
        self._orig_streams = {}
        self._isaac_initialized = False
        # per-task recorder output overrides (set right before creating env)
        self._recorder_export_dir = None
        self._recorder_filename = None
        
        # Force batch_size=1 to ensure each actor processes exactly one task
        self.batch_size = 1
        
        logger.info(
            f"Initialized AnnotateDemosMapper: task={task_name}, "
            f"device={device}, headless={headless}, batch_size=1 (one task per actor)"
        )

    def _ensure_sim_app(self):
        """Ensure Isaac Sim SimulationApp is initialized once and reused."""
        if self._isaac_initialized:
            return

        import torch
        import os
        # Clean up any existing CUDA state
        if torch.cuda.is_initialized():
            logger.warning("CUDA was initialized before Isaac Sim. Cleaning up...")
            torch.cuda.empty_cache()
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Isaac Sim requires CUDA to run. "
                "Please check your GPU drivers and CUDA installation."
            )
        # Set environment variables for CUDA
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        logger.info(f"Initializing Isaac Sim (SimulationApp) for task: {self.task_name}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Using device: {self.device}")

        # Temporarily swap std streams to real OS streams during SimulationApp launch
        import sys
        for stream_name in ('stdin', 'stdout', 'stderr'):
            self._orig_streams[stream_name] = getattr(sys, stream_name, None)
            real_stream = getattr(sys, f"__{stream_name}__", None)
            if real_stream is not None:
                setattr(sys, stream_name, real_stream)

        # Build and launch SimulationApp
        import argparse
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
        import faulthandler
        try:
            self._faulthandler_file = open(os.devnull, "w")
            faulthandler.enable(file=self._faulthandler_file)
        except Exception as e:
            logger.debug("Failed to enable faulthandler: {}", e)
            self._faulthandler_file = None

        app_launcher = AppLauncher(args)
        self._simulation_app = app_launcher.app

        # Restore original wrapped streams
        for stream_name, orig in self._orig_streams.items():
            if orig is not None:
                setattr(sys, stream_name, orig)

        # Import env packages after SimulationApp
        import isaaclab_mimic.envs  # noqa: F401
        if self.enable_pinocchio:
            import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

        self._isaac_initialized = True
        logger.info("Isaac Sim SimulationApp initialized successfully")

    def _create_task_env(self):
        """Create a fresh Isaac Lab env for a single task.

        Recorder export paths are taken from self._recorder_export_dir and
        self._recorder_filename, which must be set by the caller before
        invoking this method (keeps API consistent with other operator params).
        """
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers import RecorderTerm, RecorderTermCfg
        from isaaclab.utils import configclass

        # Parse environment config for this task
        env_cfg = parse_env_cfg(self.task_name, device=self.device, num_envs=1)
        env_cfg.env_name = self.task_name

        # Record success termination term (if exists); disable all terminations
        self._success_term = getattr(env_cfg.terminations, "success", None)
        env_cfg.terminations = None

        # Setup recorder configuration and set export paths like the original script
        env_cfg.recorders = self._create_recorder_config(
            ActionStateRecorderManagerCfg,
            RecorderTerm,
            RecorderTermCfg,
            configclass,
        )
        env_cfg.recorders.dataset_export_dir_path = self._recorder_export_dir
        env_cfg.recorders.dataset_filename = self._recorder_filename

        # Create a new env for this task
        self._env = gym.make(self.task_name, cfg=env_cfg).unwrapped
        self._env.reset()
        return self._env

    def _create_recorder_config(self, ActionStateRecorderManagerCfg, RecorderTerm, RecorderTermCfg, configclass):
        """Create recorder configuration for mimic annotations."""

        # Define custom recorder terms
        class PreStepDatagenInfoRecorder(RecorderTerm):
            """Recorder term that records the datagen info data in each step."""

            def record_pre_step(self):
                eef_pose_dict = {}
                for eef_name in self._env.cfg.subtask_configs.keys():
                    eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

                datagen_info = {
                    "object_pose": self._env.get_object_poses(),
                    "eef_pose": eef_pose_dict,
                    "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
                }
                return "obs/datagen_info", datagen_info

        @configclass
        class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
            class_type = PreStepDatagenInfoRecorder

        class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
            """Recorder term that records the subtask completion observations in each step."""

            def record_pre_step(self):
                return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()

        @configclass
        class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
            class_type = PreStepSubtaskTermsObservationsRecorder

        @configclass
        class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
            """Mimic specific recorder terms."""

            record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
            record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()

        return MimicRecorderManagerCfg()

    def _annotate_file(
        self,
        input_file: str,
        output_file: str,
    ) -> Dict[str, Any]:
        """
        Annotate entire HDF5 file with all episodes.
        
        :param input_file: Input HDF5 file path
        :param output_file: Output HDF5 file path
        :return: Annotation result
        """
        # Ensure SimulationApp is running (only once), but recreate Isaac Lab env per task
        self._ensure_sim_app()
        
        import torch
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        
        # Load dataset
        dataset_handler = HDF5DatasetFileHandler()
        dataset_handler.open(input_file)
        episode_count = dataset_handler.get_num_episodes()


        logger.info(f"Processing {episode_count} episodes from {input_file}")

        # Setup output path (use absolute path for consistency)
        output_dir = os.path.dirname(output_file)
        output_filename = os.path.splitext(os.path.basename(output_file))[0]
        # Resolve to absolute directory (without coupling to Data-Juicer's work_dir)
        output_dir_abs = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(output_dir_abs, exist_ok=True)

        # Create a fresh env for this task and configure recorder outputs
        # Close any existing env to avoid cross-task contamination
        if self._env is not None:
            self._env.close()
            self._env = None

        # Set recorder output targets for this task and create a fresh env
        self._recorder_export_dir = output_dir_abs
        self._recorder_filename = output_filename
        self._create_task_env()
        assert self._env is not None, "Environment should be created by _create_task_env()"

        # The actual output file path that Isaac will write to
        output_file_abs = os.path.join(output_dir_abs, f"{output_filename}.hdf5")

        # Process each episode
        success_count = 0
        for episode_index, episode_name in enumerate(dataset_handler.get_episode_names()):
            logger.info(f"Annotating episode #{episode_index} ({episode_name})")

            episode = dataset_handler.load_episode(episode_name, self._env.device)
            success = self._replay_and_annotate(episode)

            if success:
                # Set success and export
                self._env.recorder_manager.set_success_to_episodes(
                    None, torch.tensor([[True]], dtype=torch.bool, device=self._env.device)
                )
                self._env.recorder_manager.export_episodes()
                success_count += 1
                logger.info(f"\tExported annotated episode {episode_index}")
            else:
                logger.warning(f"\tSkipped episode {episode_index} due to annotation failure")

        dataset_handler.close()

        logger.info(
            f"Exported {success_count}/{episode_count} annotated episodes to {output_file_abs}"
        )

        # NOTE: Do NOT close env here - keep it alive for subsequent batches
        # The environment will be closed in cleanup() when the Actor shuts down
        # Closing and recreating Isaac Sim environment causes deadlocks

        return {
            "success": True,
            "output_file": output_file_abs,
            "total_episodes": episode_count,
            "successful_episodes": success_count,
        }

    def _replay_and_annotate(self, episode) -> bool:
        """
        Replay episode in environment and annotate (automatic mode).
        
        :param episode: Episode data to replay
        :return: True if annotation successful, False otherwise
        """
        assert self._env is not None, "Environment must be initialized before replay"
        import torch
        
    # Extract initial state and actions
        initial_state = episode.data["initial_state"]
        actions = episode.data["actions"]
        
        # Reset environment to initial state
        self._env.sim.reset()
        self._env.recorder_manager.reset()
        self._env.reset_to(initial_state, None, is_relative=True)
        
        # Replay all actions
        for action in actions:
            action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
            self._env.step(action_tensor)
        
        # Check if task was completed successfully
        if self._success_term is not None:
            success_result = self._success_term.func(self._env, **self._success_term.params)[0]
            if not bool(success_result):
                logger.warning("Episode replay failed: task not completed")
                return False
        
        # Verify all subtask term signals are annotated
        annotated_episode = self._env.recorder_manager.get_episode(0)
        subtask_signals = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        
        for signal_name, signal_flags in subtask_signals.items():
            if not torch.any(torch.tensor(signal_flags)):
                logger.warning(f"Subtask '{signal_name}' not completed")
                return False
        
        return True



    def process_batched(self, samples: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single annotation task (batch_size=1).
        
        After processing, the actor will exit to ensure clean Isaac Sim state.
        Ray will automatically create a new actor for the next task.
        
        Each sample should contain:
        - input_file: Path to input HDF5 file
        - output_file: Path to output annotated HDF5 file
        
        :param samples: Batch of samples (must contain exactly 1 sample)
        :param rank: Ray actor rank for GPU assignment
        :return: Processed batch with annotation results
        """
        # Ensure we only process one task per actor
        num_samples = len(samples[self.text_key])
        if num_samples != 1:
            logger.warning(
                f"AnnotateDemosMapper expects batch_size=1, got {num_samples}. "
                f"Processing only the first sample."
            )
            # Truncate to first sample only
            samples = {k: [v[0]] for k, v in samples.items()}
            num_samples = 1
        # Per-process GPU assignment: if device is 'cuda:auto', bind this process to a GPU by rank
        try:
            import os as _os
            import torch as _torch
            if isinstance(self.device, str) and self.device.startswith("cuda"):
                # Honor explicit device like 'cuda:1'; only auto-assign when 'cuda' or 'cuda:auto'
                if self.device in ("cuda", "cuda:auto"):
                    # If Ray sets CUDA_VISIBLE_DEVICES to a single GPU, pin to cuda:0 in the local view
                    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if visible and len(visible.split(",")) == 1:
                        self.device = "cuda:0"
                    else:
                        # Default executor: spread by rank across all visible GPUs
                        # Rank will be provided by the framework when _accelerator == 'cuda'
                        # Fallback to 0 if rank or device count not available
                        def _assign_device_by_rank(_rank: Optional[int]):
                            if _torch.cuda.is_available():
                                count = _torch.cuda.device_count()
                                if _rank is not None and count > 0:
                                    return f"cuda:{_rank % count}"
                                if count > 0:
                                    return "cuda:0"
                            return self.device
                        self.device = _assign_device_by_rank(rank)
        except Exception:
            # Do not fail batch processing due to device binding issues; keep existing self.device
            pass

        logger.info(f"Processing single annotation task (actor will exit after completion)")
        
        # Prepare results container for single task
        results: List[Optional[Dict[str, Any]]] = [None]
        
        # Check required keys
        if self.input_file_key not in samples or self.output_file_key not in samples:
            logger.error(
                f"Required keys '{self.input_file_key}' and/or '{self.output_file_key}' "
                f"not found in samples"
            )
            return samples
        
        # Process the single task
        input_file = samples[self.input_file_key][0]
        output_file = samples[self.output_file_key][0]
        
        logger.info(f"Task: {input_file} -> {output_file}")
        
        try:
            result = self._annotate_file(input_file, output_file)
            results[0] = result
            
            if result.get("success"):
                logger.info(f"✓ Task completed successfully")
            else:
                logger.warning(f"✗ Task failed: {result.get('error')}")
            
            samples["annotation_result"] = results
            
        finally:
            # Cleanup after processing one task
            # Note: Actor exit is handled by the framework layer, not here
            logger.info("Task complete. Cleaning up resources...")
            self.cleanup()
        
        return samples
    
    def cleanup(self):
        """Cleanup Isaac Sim resources (called by Ray Actor on shutdown).
        
        Only closes the Isaac Lab environment, not the simulation app.
        Ray will manage the Actor process lifecycle.
        """
        logger.info("Cleaning up AnnotateDemosMapper resources...")
        
        # Close Isaac Lab environment
        if self._env is not None:
            self._env.close()
            logger.info("Closed Isaac Lab environment")
            self._env = None
        
        # Do NOT close simulation_app - it calls exit() which conflicts with Ray
        # Ray will terminate the Actor process naturally
        logger.info("Cleanup complete (simulation_app left running for Ray to manage)")
        return {"status": "cleaned"}
    
    def __del__(self):
        """Cleanup Isaac Sim environment on deletion."""
        self.cleanup()
