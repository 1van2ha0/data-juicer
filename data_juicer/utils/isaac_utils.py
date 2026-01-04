import argparse
import faulthandler
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import torch
from loguru import logger

# ============================================================================
# Context Managers
# ============================================================================


@contextmanager
def LazyStreamRedirector():
    """
    Context manager to temporarily redirect sys.stdin, sys.stdout, and sys.stderr
    to the real OS streams (if available) and restore them afterwards.

    This is useful when launching applications like Isaac Sim's SimulationApp
    that might interfere with wrapped streams (e.g., from Ray or other loggers).
    """
    orig_streams: Dict[str, Optional[Any]] = {}
    try:
        # Swap wrapped IO streams with the real OS streams
        for stream_name in ("stdin", "stdout", "stderr"):
            orig_streams[stream_name] = getattr(sys, stream_name, None)
            real_stream = getattr(sys, f"__{stream_name}__", None)
            if real_stream is not None:
                setattr(sys, stream_name, real_stream)
        yield
    finally:
        # Restore wrapped streams
        for stream_name, orig_stream in orig_streams.items():
            if orig_stream is not None:
                setattr(sys, stream_name, orig_stream)


# ============================================================================
# Core Initialization
# ============================================================================


def init_isaac_sim_app(
    headless: bool = True,
    device: str = "cuda:auto",
    enable_cameras: bool = False,
    enable_pinocchio: bool = False,
) -> Any:
    """
    Initialize Isaac Sim SimulationApp with common configurations.

    :param headless: Run Isaac Sim in headless mode.
    :param device: Device to run on ('cuda:0', 'cpu', etc.).
    :param enable_cameras: Enable cameras in Isaac Sim.
    :param enable_pinocchio: Enable Pinocchio support.
    :return: The initialized SimulationApp instance.
    :raises RuntimeError: If CUDA is not available.
    :raises ImportError: If Isaac Lab is not installed.
    """
    # 1. CUDA checks and setup
    if torch.cuda.is_initialized():
        logger.warning("CUDA was initialized before Isaac Sim. Clearing cached state...")
        torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Isaac Sim requires CUDA to run. "
            "Please verify the GPU driver and CUDA installation."
        )

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    logger.info(f"Initializing Isaac Sim SimulationApp (device={device}, headless={headless})")
    logger.info(f"CUDA device count detected by torch: {torch.cuda.device_count()}")

    # 2. Prepare AppLauncher arguments
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        raise ImportError("Could not import isaaclab.app.AppLauncher. " "Please ensure Isaac Lab is installed.")

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    # Parse empty args to get defaults, then override
    args, _ = parser.parse_known_args([])
    args.headless = headless
    args.device = device
    args.enable_cameras = enable_cameras
    args.enable_scene_lights = not headless

    # 3. Pinocchio setup (pre-launch)
    if enable_pinocchio:
        import pinocchio  # noqa: F401

    # 4. Faulthandler setup
    # Redirect faulthandler to devnull to avoid noise
    try:
        faulthandler_file = open(os.devnull, "w")
        faulthandler.enable(file=faulthandler_file)
        # Note: We don't return the file handle here, relying on OS/GC to handle it eventually.
        # If strict management is needed, this function should return (app, file_handle).
    except Exception as exc:
        logger.debug(f"Failed to enable faulthandler: {exc}")

    # 5. Launch SimulationApp with stream redirection
    simulation_app = None
    with LazyStreamRedirector():
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

    # 6. Pinocchio setup (post-launch)
    if enable_pinocchio:
        import isaaclab.utils.math  # noqa: F401
        import warp  # noqa: F401

    logger.info("Isaac Sim SimulationApp initialized successfully")
    return simulation_app


# ============================================================================
# Mapper Lifecycle Helpers
# ============================================================================


def ensure_isaac_sim_app(instance: Any, mode: str = "mimic") -> None:
    """
    Ensure Isaac Sim SimulationApp is initialized once and reused.
    Handles common initialization and mode-specific imports.

    This function is designed to be called within a Mapper's processing method.
    It checks `instance._isaac_initialized` to avoid re-initialization.

    :param instance: The mapper instance (self). Expected to have attributes:
                     headless, device, enable_cameras (or video), enable_pinocchio.
    :param mode: 'mimic' for Isaac Lab Mimic, 'tasks' for Isaac Lab Tasks.
    """
    if getattr(instance, "_isaac_initialized", False):
        return

    # Determine enable_cameras
    enable_cameras = False
    if hasattr(instance, "enable_cameras"):
        enable_cameras = instance.enable_cameras
    elif hasattr(instance, "video"):
        enable_cameras = bool(instance.video)

    instance._simulation_app = init_isaac_sim_app(
        headless=instance.headless,
        device=instance.device,
        enable_cameras=enable_cameras,
        enable_pinocchio=getattr(instance, "enable_pinocchio", False),
    )

    # Mode-specific imports
    if mode == "mimic":
        import isaaclab_mimic.envs  # noqa: F401

        if getattr(instance, "enable_pinocchio", False):
            import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

    elif mode == "tasks":
        import isaaclab_tasks  # noqa: F401

        if getattr(instance, "enable_pinocchio", False):
            import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
            import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

    instance._isaac_initialized = True


def cleanup_isaac_env(instance: Any) -> Dict[str, str]:
    """
    Safely close an Isaac Lab environment stored in a mapper instance.
    Can be assigned directly to the `cleanup` method of a mapper class.

    :param instance: The mapper instance containing `_env`.
    :return: A status dictionary (e.g., {"status": "cleaned"}).
    """
    logger.info(f"Cleaning up {instance.__class__.__name__} resources...")

    if hasattr(instance, "_env") and instance._env is not None:
        try:
            instance._env.close()
            logger.info("Closed Isaac Lab environment.")
        except Exception as e:
            logger.warning(f"Error closing Isaac Lab environment: {e}")
        finally:
            instance._env = None

    # Optional: Handle faulthandler if it was stored
    if hasattr(instance, "_faulthandler_file") and instance._faulthandler_file is not None:
        try:
            import faulthandler

            faulthandler.disable()
            instance._faulthandler_file.close()
        except Exception:
            pass
        finally:
            instance._faulthandler_file = None

    logger.info("Cleanup complete (simulation_app left for Ray to manage).")
    return {"status": "cleaned"}


# ============================================================================
# Utility Functions
# ============================================================================


def resolve_nucleus_paths(config: Union[Dict, list, str]) -> Union[Dict, list, str]:
    """
    Recursively resolve Isaac Nucleus paths in a configuration dictionary or list.
    Replaces placeholders like '{ISAAC_NUCLEUS_DIR}' with actual paths.

    :param config: Configuration object (dict, list, or string).
    :return: Configuration object with resolved paths.
    """
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

    def _replace_recursive(item):
        if isinstance(item, str):
            item = item.replace("{ISAAC_NUCLEUS_DIR}", ISAAC_NUCLEUS_DIR)
            item = item.replace("{NVIDIA_NUCLEUS_DIR}", NVIDIA_NUCLEUS_DIR)
        elif isinstance(item, list):
            item = [_replace_recursive(i) for i in item]
        elif isinstance(item, dict):
            for k, v in item.items():
                item[k] = _replace_recursive(v)
        return item

    return _replace_recursive(config)


def create_video_from_images(input_pattern: str, output_video: str, framerate: float = 20.0) -> bool:
    """
    Create a video from a sequence of images using ffmpeg.

    :param input_pattern: Input file pattern (e.g., "frame_%04d.png").
    :param output_video: Output video file path.
    :param framerate: Frame rate of the output video.
    :return: True if successful, False otherwise.
    """
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
